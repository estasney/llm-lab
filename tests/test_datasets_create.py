import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from llm_lab.cli import app
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.settings import Settings, get_settings
from llm_lab.studio.client import RecipeError, RecipeValidation, StudioClient, StudioLogin, open_studio_client
from llm_lab.studio.ssh import open_ssh_session
from tests.conftest import run_cli


def build_snapshot(status: str, *, error: str | None = None, artifact_path: str | None = None) -> dict[str, object]:
    return {
        "job_id": "job-1",
        "status": status,
        "stage": None,
        "current_column": None,
        "progress": {"done": None, "total": None, "percent": None},
        "error": error,
        "artifact_path": artifact_path,
    }


@dataclass
class FakeStudio:
    statuses: list[dict[str, object]]
    validation: RecipeValidation = field(
        default_factory=lambda: RecipeValidation(valid=True, errors=[], raw_detail=None)
    )
    revoke_token_at_job_start: bool = False
    tokens_issued: int = 0
    valid_token: str = ""
    requests: list[tuple[str, str]] = field(default_factory=list)
    job_payloads: list[object] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        self.requests.append(route)
        if route == ("POST", "/api/auth/login"):
            self.tokens_issued += 1
            self.valid_token = f"token-{self.tokens_issued}"
            return httpx.Response(200, json={"access_token": self.valid_token})
        if request.headers.get("Authorization") != f"Bearer {self.valid_token}":
            return httpx.Response(401, json={"detail": "Invalid or expired token"})
        match route:
            case ("POST", "/api/data-recipe/validate"):
                return httpx.Response(200, json=self.validation.model_dump(mode="json"))
            case ("POST", "/api/data-recipe/jobs"):
                self.job_payloads.append(json.loads(request.content))
                if self.revoke_token_at_job_start:
                    self.valid_token = "revoked"
                return httpx.Response(200, json={"job_id": "job-1"})
            case ("GET", "/api/data-recipe/jobs/job-1/status"):
                return httpx.Response(200, json=self.statuses.pop(0))
            case _:
                return httpx.Response(404, json={"detail": "not found"})


@dataclass
class FakeRunDirectories:
    fetched: list[PurePosixPath] = field(default_factory=list)

    def fetch_directory(self, remote: PurePosixPath, local: Path) -> None:
        self.fetched.append(remote)
        local.mkdir()
        (local / "metadata.json").write_text("{}")


def build_app(studio: FakeStudio, run_directories: FakeRunDirectories, datasets_dir: Path) -> DependentTyper:
    http = httpx.Client(
        transport=httpx.MockTransport(studio.handle),
        base_url="http://studio",
        auth=StudioLogin("unsloth", SecretStr("password")),
    )
    return app.with_dependency_overrides(
        {
            get_settings: lambda: Settings(
                datasets_dir=datasets_dir,
                studio_ssh_host="studio.test",
                studio_ssh_port=22,
                studio_password=SecretStr("password"),
            ),
            open_studio_client: lambda: StudioClient(http, poll_seconds=0),
            open_ssh_session: lambda: run_directories,
        }
    )


@pytest.fixture
def datasets_dir(tmp_path: Path) -> Path:
    return tmp_path / "datasets"


@pytest.fixture
def recipe_path(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"recipe": {"columns": [{"name": "topic"}]}, "run": {"rows": 99}, "ui": {"nodes": []}}))
    return path


@pytest.fixture
def run_directories() -> FakeRunDirectories:
    return FakeRunDirectories()


def test_create_fetches_the_run_directory_of_a_completed_job(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a Studio whose job is active on the first poll and completed on the second.
    Act: create a dataset from a recipe file.
    Assert: the job received the recipe with the run options of the command, and the run directory is the dataset.
    """
    studio = FakeStudio([build_snapshot("active"), build_snapshot("completed", artifact_path="/runs/recipe_topics")])
    cli = build_app(studio, run_directories, datasets_dir)

    run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "topics", "--rows", "3"])

    expected_payload = {
        "recipe": {"columns": [{"name": "topic"}]},
        "run": {"rows": 3, "run_name": "topics", "execution_type": "full"},
    }
    assert studio.job_payloads == [expected_payload], f"Expected one job with {expected_payload}."
    assert run_directories.fetched == [PurePosixPath("/runs/recipe_topics")], "Expected a fetch of the job's path."
    actual_files = sorted(path.name for path in (datasets_dir / "topics").iterdir())
    assert actual_files == ["metadata.json"], f"Expected the fetched files in the dataset, got {actual_files}."
    leftovers = sorted(path.name for path in datasets_dir.iterdir() if path.name != "topics")
    assert leftovers == [], f"Expected no partial directory after the fetch, got {leftovers}."


def test_create_logs_in_again_after_studio_revokes_the_token(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a Studio that revokes the access token at job start.
    Act: create a dataset.
    Assert: the client logged in a second time and the dataset exists.
    """
    studio = FakeStudio(
        [build_snapshot("completed", artifact_path="/runs/recipe_topics")], revoke_token_at_job_start=True
    )
    cli = build_app(studio, run_directories, datasets_dir)

    run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "topics", "--rows", "3"])

    assert studio.tokens_issued == 2, f"Expected two logins, got {studio.tokens_issued}."
    assert (datasets_dir / "topics").is_dir(), "Expected the dataset directory after the second login."


def test_create_raises_the_error_of_a_failed_job(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a Studio whose job ends with an error.
    Act: create a dataset.
    Assert: the command raised the job's error and fetched nothing.
    """
    studio = FakeStudio([build_snapshot("error", error="model not loaded")])
    cli = build_app(studio, run_directories, datasets_dir)

    with pytest.raises(RuntimeError, match="model not loaded"):
        run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "topics", "--rows", "3"])

    assert run_directories.fetched == [], "Expected no fetch for a failed job."
    assert not (datasets_dir / "topics").exists(), "Expected no dataset directory for a failed job."


def test_create_starts_no_job_for_a_recipe_studio_rejects(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a Studio whose validation reports an error.
    Act: create a dataset.
    Assert: the command raised the validation message and started no job.
    """
    rejection = RecipeValidation(
        valid=False,
        errors=[RecipeError(message="unknown column type", path="columns.0", code=None)],
        raw_detail=None,
    )
    studio = FakeStudio([], validation=rejection)
    cli = build_app(studio, run_directories, datasets_dir)

    with pytest.raises(ValueError, match="unknown column type"):
        run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "topics", "--rows", "3"])

    assert studio.job_payloads == [], "Expected no job for a rejected recipe."


def test_create_sends_nothing_when_the_dataset_exists(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a datasets directory that already holds the dataset name.
    Act: create a dataset with that name.
    Assert: the command raised FileExistsError and Studio received no request.
    """
    (datasets_dir / "topics").mkdir(parents=True)
    studio = FakeStudio([])
    cli = build_app(studio, run_directories, datasets_dir)

    with pytest.raises(FileExistsError, match="topics"):
        run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "topics", "--rows", "3"])

    assert studio.requests == [], f"Expected no request to Studio, got {studio.requests}."


def test_create_rejects_a_name_that_is_not_a_plain_directory_name(
    datasets_dir: Path, recipe_path: Path, run_directories: FakeRunDirectories
) -> None:
    """
    Arrange: a dataset name with a parent-directory segment.
    Act: create a dataset with that name.
    Assert: the command raised a validation error and Studio received no request.
    """
    studio = FakeStudio([])
    cli = build_app(studio, run_directories, datasets_dir)

    with pytest.raises(ValidationError, match="run_name"):
        run_cli(cli, ["datasets", "create", str(recipe_path), "--name", "../topics", "--rows", "3"])

    assert studio.requests == [], f"Expected no request to Studio, got {studio.requests}."

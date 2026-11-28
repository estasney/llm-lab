import json
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.seed import SamplingStrategy, SeedConfig

from llm_lab.datasets.designer import DataDesignerRunner
from llm_lab.datasets.recipe import Recipe, parse_recipe
from llm_lab.errors import CommandError
from llm_lab.keyring_settings import SecretName
from llm_lab.secrets import SettingsSecretSource
from llm_lab.settings import Settings
from tests.conftest import build_model_config, build_recipe_source


@dataclass
class ChatServer:
    port: int
    models: list[str] = field(default_factory=list)


def serve_chat_completions(server: ChatServer) -> type[BaseHTTPRequestHandler]:
    class ChatCompletionHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.models.append(request["model"])
            body = json.dumps(
                {
                    "id": "chat",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A generated sentence."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ChatCompletionHandler


@pytest.fixture
def chat_server() -> Generator[ChatServer]:
    server = ChatServer(port=0)
    with ThreadingHTTPServer(("127.0.0.1", 0), serve_chat_completions(server)) as http:
        server.port = http.server_address[1]
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        yield server
        http.shutdown()
        thread.join()


def build_runner(tmp_path: Path, port: int) -> DataDesignerRunner:
    return DataDesignerRunner(Settings(artifacts_dir=tmp_path / "artifacts", vllm_port=port))


def test_validate_accepts_a_recipe_whose_prompts_name_its_columns(tmp_path: Path, recipe: Recipe) -> None:
    build_runner(tmp_path, 8000).validate(recipe)


def test_validate_rejects_a_prompt_that_names_a_missing_column(tmp_path: Path) -> None:
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], prompt="Write about {{ season }}."))

    with pytest.raises(CommandError, match="Data Designer rejected the recipe"):
        build_runner(tmp_path, 8000).validate(recipe)


def test_create_generates_the_rows_with_the_served_model(
    tmp_path: Path, recipe: Recipe, chat_server: ChatServer
) -> None:
    directory = build_runner(tmp_path, chat_server.port).create(recipe, "sentences", 4)

    dataset = pd.read_parquet(directory / "parquet-files")
    actual = (directory.parent, len(dataset), set(dataset["sentence"].tolist()), set(chat_server.models))
    expected = (tmp_path / "artifacts", 4, {"A generated sentence."}, {"Qwen/Qwen3-8B"})
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_create_writes_one_file_for_each_batch_of_the_recipe(tmp_path: Path, chat_server: ChatServer) -> None:
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], rows_per_batch=2))

    directory = build_runner(tmp_path, chat_server.port).create(recipe, "sentences", 4)

    actual = len(list((directory / "parquet-files").glob("*.parquet")))
    assert actual == 2, f"Expected two batch files for four rows, got {actual}."


def test_config_reads_the_hub_seed_of_the_recipe_with_the_hub_token(tmp_path: Path) -> None:
    seed = {"kind": "hub", "path": "buckets/lab/seeds/0a1b/*.parquet", "sampling_strategy": "shuffle"}
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], seed=seed))

    actual = build_runner(tmp_path, 8000).build_config(recipe).get_seed_config()

    expected = SeedConfig.model_validate(
        {
            "source": {"seed_type": "hf", "path": "buckets/lab/seeds/0a1b/*.parquet", "token": SecretName.hf_token},
            "sampling_strategy": SamplingStrategy.SHUFFLE,
        }
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_designer_resolves_the_hub_token_that_a_job_receives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_LAB_HF_TOKEN", raising=False)
    monkeypatch.setenv(SecretName.hf_token, "hf_secret")

    actual = SettingsSecretSource(Settings()).resolve(SecretName.hf_token)

    assert actual == "hf_secret", f"Expected the token from HF_TOKEN, got {actual!r}."


def test_create_generates_one_row_for_each_local_seed_row(tmp_path: Path, chat_server: ChatServer) -> None:
    seeds = tmp_path / "topics.parquet"
    pd.DataFrame({"season": ["winter", "summer"]}).to_parquet(seeds)
    source = build_recipe_source(
        [build_model_config("writer")],
        prompt="Write one sentence about {{ topic }} in {{ season }}.",
        seed={"kind": "local", "path": str(seeds)},
    )

    directory = build_runner(tmp_path, chat_server.port).create(parse_recipe(source), "sentences", 4)

    dataset = pd.read_parquet(directory / "parquet-files")
    actual = dataset["season"].tolist()
    assert actual == ["winter", "summer", "winter", "summer"], f"Expected the seed rows in order, got {actual}."

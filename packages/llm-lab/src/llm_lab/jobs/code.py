import hashlib
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, ValidationError

from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import JobCode
from llm_lab.settings import Settings


class LockedSource(BaseModel):
    editable: str | None = None
    virtual: str | None = None


class LockedDependency(BaseModel):
    name: str


class LockedPackage(BaseModel):
    name: str
    source: LockedSource
    dependencies: tuple[LockedDependency, ...] = ()


class LockManifest(BaseModel):
    members: tuple[str, ...]


class WorkspaceLock(BaseModel):
    manifest: LockManifest
    package: tuple[LockedPackage, ...]


@dataclass(frozen=True)
class MemberSelection:
    project_files: tuple[str, ...]
    directories: tuple[str, ...]


def locate_member(package: LockedPackage) -> PurePosixPath:
    match package.source:
        case (
            LockedSource(editable=str() as directory, virtual=None)
            | LockedSource(editable=None, virtual=str() as directory)
        ):
            return PurePosixPath(directory)
        case _:
            raise ValueError(f"the lock file gives no directory for the workspace member {package.name}")


def select_members(lock: WorkspaceLock, package: str) -> MemberSelection:
    members = {locked.name: locked for locked in lock.package if locked.name in lock.manifest.members}
    if package not in members:
        raise CommandError(
            f"The workspace has no member with the name {package}. Its members are {sorted(members)}. "
            "Run `uv lock` after you add a member."
        )
    selected: dict[str, LockedPackage] = {}
    pending = [package]
    while pending:
        name = pending.pop()
        if name not in selected:
            selected[name] = members[name]
            pending.extend(dependency.name for dependency in members[name].dependencies if dependency.name in members)
    return MemberSelection(
        project_files=tuple(
            sorted((locate_member(member) / "pyproject.toml").as_posix() for member in members.values())
        ),
        directories=tuple(sorted(locate_member(member).as_posix() for member in selected.values())),
    )


class WorkspaceCodeSource:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.workspace_dir

    def read_lock(self) -> WorkspaceLock:
        lock_file = self.root / "uv.lock"
        try:
            return WorkspaceLock.model_validate(tomllib.loads(lock_file.read_text()))
        except FileNotFoundError as error:
            raise CommandError(
                f"{lock_file} does not exist. Run llm-lab in the root directory of the uv workspace, "
                "or set LLM_LAB_WORKSPACE_DIR."
            ) from error
        except ValidationError as error:
            raise CommandError(f"{lock_file} is not the lock file of a uv workspace: {error}") from error

    def list_files(self, selection: MemberSelection) -> tuple[str, ...]:
        command = [
            "git",
            "-C",
            str(self.root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "pyproject.toml",
            "uv.lock",
            *selection.project_files,
            *selection.directories,
        ]
        try:
            listing = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise CommandError("git is not installed. llm-lab uses git to list the code files of a job.") from error
        if listing.returncode != 0:
            raise CommandError(f"git could not list the files of {self.root}: {listing.stderr.strip()}")
        return tuple(sorted({path for path in listing.stdout.split("\0") if path and (self.root / path).is_file()}))

    def read(self, package: str) -> JobCode:
        files = self.list_files(select_members(self.read_lock(), package))
        listing = hashlib.sha256()
        for path in files:
            with (self.root / path).open("rb") as file:
                listing.update(f"{path}\0{hashlib.file_digest(file, 'sha256').hexdigest()}\n".encode())
        return JobCode(root=self.root, files=files, digest=listing.hexdigest())

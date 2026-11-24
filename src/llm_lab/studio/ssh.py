import shlex
import subprocess
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Annotated, Protocol, assert_never

from llm_lab.dependencies import Depends
from llm_lab.settings import StudioSettingsDep


class RunDirectorySource(Protocol):
    def fetch_directory(self, remote: PurePosixPath, local: Path) -> None: ...


@dataclass(frozen=True)
class SshSession:
    destination: str
    control_socket: Path
    studio_socket: Path

    def fetch_directory(self, remote: PurePosixPath, local: Path) -> None:
        subprocess.run(
            [
                "rsync",
                "--archive",
                "--rsh",
                shlex.join(["ssh", "-S", str(self.control_socket)]),
                f"{self.destination}:{remote}/",
                f"{local}/",
            ],
            check=True,
        )


def open_ssh_session(settings: StudioSettingsDep) -> Generator[SshSession]:
    match settings.ssh_identity_file:
        case None:
            identity_options = []
        case Path() as identity_file:
            identity_options = ["-i", str(identity_file), "-o", "IdentitiesOnly=yes"]
        case _:
            assert_never(settings.ssh_identity_file)
    settings.known_hosts_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        session = SshSession(
            destination=f"{settings.ssh_user}@{settings.ssh_host}",
            control_socket=Path(directory) / "control",
            studio_socket=Path(directory) / "studio",
        )
        subprocess.run(
            [
                "ssh",
                "-M",
                "-S",
                str(session.control_socket),
                "-f",
                "-N",
                "-L",
                f"{session.studio_socket}:127.0.0.1:{settings.port}",
                "-p",
                str(settings.ssh_port),
                "-o",
                "LogLevel=ERROR",
                "-o",
                "BatchMode=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={settings.known_hosts_file}",
                *identity_options,
                session.destination,
            ],
            check=True,
        )
        try:
            yield session
        finally:
            subprocess.run(
                ["ssh", "-q", "-S", str(session.control_socket), "-O", "exit", session.destination], check=True
            )


type SshSessionDep = Annotated[SshSession, Depends(open_ssh_session)]
type RunDirectorySourceDep = Annotated[RunDirectorySource, Depends(open_ssh_session)]

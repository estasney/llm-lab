import shlex
import subprocess
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Annotated, Protocol, assert_never

from llm_lab.dependencies import Depends
from llm_lab.settings import SettingsDep


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
                shlex.join(["ssh", f"-oControlPath={self.control_socket}"]),
                f"{self.destination}:{remote}/",
                f"{local}/",
            ],
            check=True,
        )


def open_ssh_session(settings: SettingsDep) -> Generator[SshSession]:
    settings.studio_known_hosts_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        session = SshSession(
            destination=f"{settings.studio_ssh_user}@{settings.studio_ssh_host}",
            control_socket=Path(directory) / "control",
            studio_socket=Path(directory) / "studio",
        )
        options = {
            "ControlMaster": "yes",
            "ControlPath": str(session.control_socket),
            "ForkAfterAuthentication": "yes",
            "SessionType": "none",
            "LocalForward": f"{session.studio_socket} 127.0.0.1:{settings.studio_port}",
            "ExitOnForwardFailure": "yes",
            "Port": str(settings.studio_ssh_port),
            "BatchMode": "yes",
            "StrictHostKeyChecking": "accept-new",
            "UserKnownHostsFile": str(settings.studio_known_hosts_file),
            "LogLevel": "ERROR",
        }
        match settings.studio_ssh_identity_file:
            case None:
                pass
            case Path() as identity_file:
                options |= {"IdentityFile": str(identity_file), "IdentitiesOnly": "yes"}
            case _:
                assert_never(settings.studio_ssh_identity_file)
        subprocess.run(
            ["ssh", *(f"-o{name}={value}" for name, value in options.items()), session.destination], check=True
        )
        try:
            yield session
        finally:
            subprocess.run(
                [
                    "ssh",
                    f"-oControlPath={session.control_socket}",
                    "-oLogLevel=QUIET",
                    "-O",
                    "exit",
                    session.destination,
                ],
                check=True,
            )


type SshSessionDep = Annotated[SshSession, Depends(open_ssh_session)]
type RunDirectorySourceDep = Annotated[RunDirectorySource, Depends(open_ssh_session)]

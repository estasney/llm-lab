import math
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from pydantic import SecretStr

from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import JobCode
from llm_lab.settings import Settings


@dataclass(frozen=True)
class KeyPair:
    private_key: SecretStr
    public_key: str


@dataclass(frozen=True)
class InstanceAccess:
    authorized_key: str
    host_private_key: SecretStr


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    port: int


class CodeDelivery(StrEnum):
    delivered = "delivered"
    unreachable = "unreachable"


def generate_key_pair() -> KeyPair:
    key = Ed25519PrivateKey.generate()
    return KeyPair(
        private_key=SecretStr(key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()).decode()),
        public_key=key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode(),
    )


class JobAccessStore:
    def __init__(self, settings: Settings) -> None:
        self.directory = settings.job_access_dir

    def create(self, job_name: str) -> InstanceAccess:
        client = generate_key_pair()
        host = generate_key_pair()
        self.remove(job_name)
        directory = self.directory / job_name
        directory.mkdir(mode=0o700, parents=True)
        identity = directory / "id_ed25519"
        identity.touch(mode=0o600)
        identity.write_text(client.private_key.get_secret_value())
        (directory / "known_hosts").write_text(f"llm-lab-{job_name} {host.public_key}\n")
        return InstanceAccess(authorized_key=client.public_key, host_private_key=host.private_key)

    def remove(self, job_name: str) -> None:
        if (self.directory / job_name).exists():
            shutil.rmtree(self.directory / job_name)

    def build_ssh_command(self, job_name: str, endpoint: SshEndpoint) -> list[str]:
        directory = self.directory / job_name
        if not directory.exists():
            raise CommandError(f"No SSH key exists for job {job_name}. Only a running job has one.")
        return [
            "ssh",
            "-p",
            str(endpoint.port),
            "-i",
            str(directory / "id_ed25519"),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"UserKnownHostsFile={directory / 'known_hosts'}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"HostKeyAlias=llm-lab-{job_name}",
            "-l",
            "root",
            endpoint.host,
        ]


class RsyncDownloader:
    def __init__(self, settings: Settings, access: JobAccessStore) -> None:
        self.directory = settings.job_downloads_dir
        self.access = access

    def download(self, job_name: str, endpoint: SshEndpoint, remote_directory: str) -> Path:
        *ssh, host = self.access.build_ssh_command(job_name, endpoint)
        local_directory = self.directory / job_name
        local_directory.mkdir(parents=True, exist_ok=True)
        command = [
            "rsync",
            "--archive",
            "--partial",
            "--rsh",
            shlex.join(ssh),
            f"{host}:{remote_directory.rstrip('/')}/",
            f"{local_directory}/",
        ]
        try:
            status = subprocess.run(command, check=False).returncode
        except FileNotFoundError as error:
            raise CommandError(
                "rsync is not installed. Install rsync, or destroy the job with --no-download."
            ) from error
        if status != 0:
            raise CommandError(
                f"rsync exited with status {status}, so the instance of job {job_name} still runs. "
                f"Run the command again, or destroy the job with --no-download."
            )
        return local_directory


class RsyncUploader:
    def __init__(self, settings: Settings, access: JobAccessStore) -> None:
        self.settings = settings
        self.access = access

    def run(self, command: list[str], input: str | None) -> CodeDelivery:
        try:
            status = subprocess.run(command, check=False, input=input, text=True).returncode
        except FileNotFoundError as error:
            raise CommandError(
                f"{command[0]} is not installed. llm-lab needs it to upload the code of a job."
            ) from error
        match status:
            case 0:
                return CodeDelivery.delivered
            case 255:
                return CodeDelivery.unreachable
            case _:
                raise CommandError(f"{command[0]} exited with status {status} during the upload of the job code.")

    def upload(self, job_name: str, endpoint: SshEndpoint, code: JobCode) -> CodeDelivery:
        *ssh, host = self.access.build_ssh_command(job_name, endpoint)
        ssh += ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={math.ceil(self.settings.job_code_poll_seconds)}"]
        directory = self.settings.job_code_directory.rstrip("/")
        copy = [
            "rsync",
            "--archive",
            "--compress",
            "--from0",
            "--files-from=-",
            "--rsh",
            shlex.join(ssh),
            f"{code.root}/",
            f"{host}:{directory}/",
        ]
        match self.run(copy, "\0".join(code.files)):
            case CodeDelivery.delivered:
                return self.run([*ssh, host, shlex.join(["touch", self.settings.job_code_marker])], None)
            case CodeDelivery.unreachable:
                return CodeDelivery.unreachable
            case _ as delivery:
                assert_never(delivery)

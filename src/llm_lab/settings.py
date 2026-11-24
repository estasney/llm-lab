from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_lab.dependencies import Depends


def default_datasets_dir() -> Path:
    return Path.home() / ".local" / "share" / "llm-lab" / "datasets"


def default_known_hosts_file() -> Path:
    return Path.home() / ".local" / "share" / "llm-lab" / "known_hosts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    datasets_dir: Path = Field(
        default_factory=default_datasets_dir, description="Directory that holds one directory per dataset."
    )
    studio_ssh_host: str = Field(description="Host name or address of the machine that runs the Studio container.")
    studio_ssh_port: int = Field(description="Port of the sshd in the Studio container.")
    studio_ssh_user: str = Field(default="root", description="User of the SSH login.")
    studio_ssh_identity_file: Path | None = Field(
        default=None, description="Private key of the SSH login. ssh chooses the key when this is unset."
    )
    studio_known_hosts_file: Path = Field(
        default_factory=default_known_hosts_file, description="File where ssh records the host keys of Studio machines."
    )
    studio_port: int = Field(default=8000, description="Port that Studio listens on inside the container.")
    studio_username: str = Field(default="unsloth", description="Name of the Studio account.")
    studio_password: SecretStr = Field(description="Password of the Studio account.")
    studio_request_timeout_seconds: float = Field(default=60, description="Time limit of one request to Studio.")
    studio_poll_seconds: float = Field(default=5, description="Wait between two reads of a job's status.")


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]


type SettingsDep = Annotated[Settings, Depends(get_settings)]

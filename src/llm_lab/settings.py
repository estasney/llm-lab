import os
from functools import cache, cached_property
from pathlib import Path
from typing import Annotated

from pydantic import Field, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from sqlalchemy import URL, Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry

from llm_lab.dependencies import Depends
from llm_lab.keyring_settings import KeyringEntry, SecretName, SecretSettingsSource, read_keyring_password


def apply_sqlite_pragmas(dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    job_database_path: Path = Field(
        default=Path(".llm-lab/jobs.sqlite3"),
        description="Path of the SQLite database file that holds the state transitions of each job.",
        examples=[".llm-lab/jobs.sqlite3", "/var/lib/llm-lab/jobs.sqlite3"],
    )
    hf_token: Annotated[SecretStr | None, KeyringEntry(service="llm-lab", username=SecretName.hf_token)] = Field(
        default=None,
        description="Hugging Face access token that a job receives as HF_TOKEN when its manifest lists HF_TOKEN in "
        "secret_environment. Set it with `llm-lab secrets set HF_TOKEN`.",
    )
    vast_api_key: Annotated[SecretStr | None, KeyringEntry(service="llm-lab", username=SecretName.vast_api_key)] = (
        Field(
            default=None,
            description="Vast.ai account API key that each request to the Vast.ai API sends. "
            "Set it with `llm-lab secrets set VAST_API_KEY`.",
        )
    )
    vast_api_url: str = Field(
        default="https://console.vast.ai/api",
        description="Base address of the Vast.ai API, without the version path.",
        examples=["https://console.vast.ai/api"],
    )
    vast_request_timeout_seconds: PositiveFloat = Field(
        default=60, description="Time limit in seconds for one HTTP request to Vast.ai.", examples=[60, 15.5]
    )
    vast_request_attempts: PositiveInt = Field(
        default=5,
        description="Maximum number of sends of one request when Vast.ai returns a rate limit or a server error.",
        examples=[5],
    )
    vast_retry_delay_seconds: PositiveFloat = Field(
        default=2,
        description="Delay in seconds before the second send of a request. "
        "Each later send multiplies the delay by the number of sends made.",
        examples=[2, 0.5],
    )
    vast_offer_limit: PositiveInt = Field(
        default=10,
        description="Maximum number of offers that one search returns. A job tries them in order of price.",
        examples=[10],
    )
    vast_log_tail_lines: PositiveInt = Field(
        default=200,
        description="Number of last log lines that are read from an instance to find the exit code of the job.",
        examples=[200],
    )
    vast_ssh_port: PositiveInt = Field(
        default=2222,
        description="Port of the SSH server in the container of a job. Vast.ai maps it to a public port.",
        examples=[2222],
    )
    job_access_dir: Path = Field(
        default=Path(".llm-lab/access"),
        description="Directory that holds the throwaway SSH key and the pinned host key of each running job.",
        examples=[".llm-lab/access"],
    )
    job_downloads_dir: Path = Field(
        default=Path(".llm-lab/downloads"),
        description="Directory where `llm-lab jobs destroy` puts the files that it downloads from each job.",
        examples=[".llm-lab/downloads"],
    )
    datasets_image: str | None = Field(
        default=None,
        description="Reference of the published datasets image that a dataset job runs.",
        examples=["ghcr.io/example/llm-lab-datasets:0.1.2"],
    )
    seed_bucket_name: str = Field(
        default="llm-lab-seeds",
        description="Name of the Hub bucket that receives the local seed files of a recipe. "
        "The bucket is in the namespace of the dataset.",
        examples=["llm-lab-seeds"],
    )
    artifacts_dir: Path = Field(
        default=Path(".llm-lab/artifacts"),
        description="Directory where Data Designer writes the generated datasets.",
        examples=[".llm-lab/artifacts", "/workspace/artifacts"],
    )
    vllm_port: PositiveInt = Field(
        default=8000, description="Port of the vLLM server on the loopback interface.", examples=[8000]
    )
    vllm_startup_timeout_seconds: PositiveFloat = Field(
        default=3600,
        description="Time limit in seconds for vLLM to download and load the model.",
        examples=[3600],
    )
    vllm_poll_seconds: PositiveFloat = Field(
        default=5, description="Delay in seconds between two health checks of vLLM.", examples=[5, 0.5]
    )
    vllm_stop_timeout_seconds: PositiveFloat = Field(
        default=60,
        description="Time in seconds that vLLM gets to stop before it is killed.",
        examples=[60],
    )

    @property
    def vllm_base_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            SecretSettingsSource(settings_cls, lambda entry: os.environ.get(entry.username)),
            SecretSettingsSource(settings_cls, read_keyring_password),
        )

    @cached_property
    def database_engine(self) -> Engine:
        self.job_database_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(URL.create("sqlite", database=str(self.job_database_path)))
        event.listen(engine, "connect", apply_sqlite_pragmas)
        return engine


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]

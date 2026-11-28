from functools import cache

from pydantic import Field, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

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
    vllm_api_key: SecretStr = Field(
        default=SecretStr("NOT_REQUIRED"),
        description="API key for vLLM.",
    )
    vllm_request_timeout_seconds: PositiveFloat = Field(
        default=600,
        description="Time limit in seconds for one chat completion request to vLLM.",
        examples=[600],
    )
    vllm_stop_timeout_seconds: PositiveFloat = Field(
        default=60,
        description="Time in seconds that vLLM gets to stop before it is killed.",
        examples=[60],
    )

    @property
    def vllm_base_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"


@cache
def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings()

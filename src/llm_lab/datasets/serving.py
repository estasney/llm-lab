import subprocess
import time
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

import httpx

from llm_lab.datasets.recipe import Recipe
from llm_lab.settings import Settings


class ModelServer(Protocol):
    def serve(self, recipe: Recipe) -> AbstractContextManager[None]: ...


def build_vllm_command(recipe: Recipe, port: int) -> list[str]:
    return ["vllm", "serve", recipe.served_model, "--host", "127.0.0.1", "--port", str(port), *recipe.vllm_args]


def is_healthy(http: httpx.Client) -> bool:
    try:
        return http.get("/health").status_code == httpx.codes.OK
    except httpx.TransportError:
        return False


class VllmServer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def wait_until_ready(self, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self.settings.vllm_startup_timeout_seconds
        with httpx.Client(base_url=self.settings.vllm_base_url, timeout=self.settings.vllm_poll_seconds) as http:
            while (status := process.poll()) is None:
                if is_healthy(http):
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError(f"vLLM was not ready after {self.settings.vllm_startup_timeout_seconds} seconds")
                time.sleep(self.settings.vllm_poll_seconds)
        raise RuntimeError(f"vLLM exited with status {status} before it was ready")

    @contextmanager
    def serve(self, recipe: Recipe) -> Generator[None]:
        with subprocess.Popen(build_vllm_command(recipe, self.settings.vllm_port)) as process:
            try:
                self.wait_until_ready(process)
                yield
            finally:
                process.terminate()
                try:
                    process.wait(timeout=self.settings.vllm_stop_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()

import subprocess
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import httpx

from llm_lab_runtime.settings import RuntimeSettings


def build_vllm_command(model: str, vllm_args: Sequence[str], port: int) -> list[str]:
    return ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port), *vllm_args]


def is_healthy(http: httpx.Client) -> bool:
    try:
        return http.get("/health").status_code == httpx.codes.OK
    except httpx.TransportError:
        return False


class VllmServer:
    def __init__(self, settings: RuntimeSettings) -> None:
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
    def serve(self, model: str, vllm_args: Sequence[str] = ()) -> Generator[None]:
        with subprocess.Popen(build_vllm_command(model, vllm_args, self.settings.vllm_port)) as process:
            try:
                self.wait_until_ready(process)
                yield
            finally:
                process.terminate()
                try:
                    process.wait(timeout=self.settings.vllm_stop_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()

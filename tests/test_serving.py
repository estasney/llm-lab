import socket
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from llm_lab.datasets.recipe import Recipe
from llm_lab.datasets.serving import VllmServer, build_vllm_command
from llm_lab.settings import Settings


def install_fake_vllm(directory: Path, monkeypatch: pytest.MonkeyPatch, program: str) -> None:
    executable = directory / "vllm"
    executable.write_text(f"#!{sys.executable}\n{textwrap.dedent(program)}")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}:{Path(sys.executable).parent}")


def find_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_server(port: int, startup_timeout_seconds: float = 30) -> VllmServer:
    return VllmServer(
        Settings(
            vllm_port=port,
            vllm_startup_timeout_seconds=startup_timeout_seconds,
            vllm_poll_seconds=0.05,
            vllm_stop_timeout_seconds=5,
        )
    )


def test_command_serves_the_model_on_the_loopback_port_with_the_recipe_arguments_last(recipe: Recipe) -> None:
    actual = build_vllm_command(recipe, 8123)

    expected = [
        "vllm",
        "serve",
        "Qwen/Qwen3-8B",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--tensor-parallel-size",
        "2",
    ]
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_serve_waits_for_health_and_stops_the_server_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: Recipe
) -> None:
    healthy_server = """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

    HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Health).serve_forever()
    """
    install_fake_vllm(tmp_path, monkeypatch, healthy_server)
    port = find_free_port()

    with build_server(port).serve(recipe):
        status_inside = httpx.get(f"http://127.0.0.1:{port}/health").status_code

    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/health")
    assert status_inside == httpx.codes.OK, f"Expected a healthy server inside the block, got {status_inside}."


def test_serve_raises_the_exit_status_of_a_server_that_stops_while_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: Recipe
) -> None:
    install_fake_vllm(tmp_path, monkeypatch, "raise SystemExit(3)")

    with pytest.raises(RuntimeError, match="status 3"), build_server(find_free_port()).serve(recipe):
        pytest.fail("Expected serve to raise before the block runs.")


def test_serve_stops_a_server_that_is_not_healthy_in_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: Recipe
) -> None:
    pid_file = tmp_path / "pid"
    install_fake_vllm(
        tmp_path, monkeypatch, f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)"
    )

    with pytest.raises(TimeoutError), build_server(find_free_port(), startup_timeout_seconds=0.5).serve(recipe):
        pytest.fail("Expected serve to raise before the block runs.")

    assert not Path(f"/proc/{pid_file.read_text()}").exists(), "Expected the server process to be stopped."

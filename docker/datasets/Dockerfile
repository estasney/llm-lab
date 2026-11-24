FROM python:3.12-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /opt/vllm
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=docker/datasets/vllm/uv.lock,target=uv.lock \
    --mount=type=bind,source=docker/datasets/vllm/pyproject.toml,target=pyproject.toml \
    uv sync --locked

WORKDIR /opt/llm-lab
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

FROM python:3.12-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/vllm/.venv /opt/vllm/.venv
COPY --from=builder /opt/llm-lab/.venv /opt/llm-lab/.venv
COPY --chmod=755 docker/datasets/dataset-job.sh /usr/local/bin/llm-lab-dataset-job

ENV PATH=/opt/llm-lab/.venv/bin:/opt/vllm/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

FROM ghcr.io/astral-sh/uv:0.13.9-python3.12-trixie-slim@sha256:b12436aceed37b21524b327362bd67e73a284b5d52f4f6be0e047dd8e4c4eb90

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/llm-lab/.venv/bin:/opt/vllm/.venv/bin:$PATH

WORKDIR /opt/vllm
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=docker/datasets/vllm/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=docker/datasets/vllm/uv.lock,target=uv.lock \
    uv sync --locked

WORKDIR /opt/llm-lab
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

COPY --chmod=755 docker/datasets/dataset-job.sh /usr/local/bin/llm-lab-dataset-job

WORKDIR /workspace
ENTRYPOINT []
CMD []

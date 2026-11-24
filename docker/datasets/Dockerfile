FROM vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e

ENV UV_PROJECT_ENVIRONMENT=/opt/llm-lab/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /opt/llm-lab/src

RUN --mount=from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d,source=/uv,target=/opt/uv/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    env -u UV_OVERRIDE -u UV_INDEX_STRATEGY UV_CACHE_DIR=/root/.cache/uv \
    /opt/uv/bin/uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=from=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d,source=/uv,target=/opt/uv/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    env -u UV_OVERRIDE -u UV_INDEX_STRATEGY UV_CACHE_DIR=/root/.cache/uv \
    /opt/uv/bin/uv sync --locked --no-dev --no-editable \
 && ln -s /opt/llm-lab/venv/bin/llm-lab /usr/local/bin/llm-lab

COPY --chmod=755 docker/datasets/dataset-job.sh /usr/local/bin/llm-lab-dataset-job

WORKDIR /workspace
ENTRYPOINT []

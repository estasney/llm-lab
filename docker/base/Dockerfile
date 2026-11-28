ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d
ARG PYTHON_VERSION=3.12
ARG VLLM_VERSION=0.33.0
ARG TORCH_BACKEND=cu130

FROM ${UV_IMAGE} AS uv

# vLLM compiles GPU kernels at runtime with nvcc, so the image keeps the CUDA toolkit.
FROM ${CUDA_IMAGE}
ARG PYTHON_VERSION
ARG VLLM_VERSION
ARG TORCH_BACKEND

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ libc6-dev libnuma-dev numactl \
    curl openssh-server rsync tmux neovim htop nvtop less procps iproute2 lsof jq ncdu tree git bash-completion \
 && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
 && echo 'export LANG=C.UTF-8' > /etc/profile.d/llm-lab-locale.sh

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed

# A job installs its locked dependencies with uv when it starts, so the image keeps uv and one Python.
COPY --from=uv /uv /uvx /bin/
RUN uv python install "${PYTHON_VERSION}"

# The vLLM environment is about 20 GB.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/vllm \
 && uv pip install --python /opt/vllm --torch-backend "${TORCH_BACKEND}" "vllm==${VLLM_VERSION}"

RUN /opt/vllm/bin/flashinfer download-kernels

COPY --chmod=755 docker/base/entrypoint.sh /usr/local/bin/llm-lab-entrypoint

ENV PATH=/opt/vllm/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda

WORKDIR /workspace
ENTRYPOINT ["llm-lab-entrypoint"]
CMD ["bash"]

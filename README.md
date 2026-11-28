# llm-lab

llm-lab runs jobs on rented Vast.ai GPU instances and manages their lifecycle. A job needs only a Docker image and a command. All other steps, such as model serving, input download and output upload, are code of the job.

## Workspace

The repository is a uv workspace with three kinds of member:

- `packages/llm-lab`: the command-line tool that runs on your machine.
- `packages/llm-lab-runtime`: code that jobs share on the instance, such as vLLM serving and the chat model for graph nodes.
- `jobs/<name>`: one member for each job. A job member declares its own dependencies.

All members resolve in one `uv.lock`. Run `uv lock` after you add a member or change a dependency. If two jobs need versions of one dependency that are not compatible, declare the conflict in `tool.uv.conflicts`.

## Job manifest

A job manifest is a YAML file with these fields:

- `name`: the job name.
- `image`: the Docker image.
- `command`: the command and its arguments.
- `package`: optional name of the workspace member that holds the code of the job.
- `environment` and `secret_environment`: optional environment variables and names of secrets.
- `disk_gb`, `ttl_seconds_after_finished` and `placement`: the disk size, the time that the instance stays after the command ends, and the GPU request or the offer.
- `artifacts_directory`: optional directory on the instance that `llm-lab jobs destroy` downloads.

Without `package`, the command runs as written and the image must hold all code.

## Jobs with a package

When the manifest has a `package`, `llm-lab jobs submit` does these steps:

1. It reads `uv.lock` and selects the member and the workspace members that it depends on.
2. It rents the instance and waits until the instance accepts SSH connections.
3. It uploads the selected members, the project file of each member, the root `pyproject.toml` and `uv.lock` with rsync to `/workspace/code`. It uploads only the files that git tracks or does not ignore.
4. The instance runs `uv run --locked --package <package> -- <command>` in `/workspace/code`. uv installs the locked dependencies of the package before the command starts.

A change to the code of a job needs no new image. The job record stores a SHA-256 digest of the uploaded files as `code_digest`.

If the code does not reach the instance in `LLM_LAB_JOB_CODE_TIMEOUT_SECONDS` (default 900), `jobs submit` destroys the instance and records the job as aborted. If your machine stops during the wait, the instance reports the exit code 124 after the same time and destroys itself after `ttl_seconds_after_finished`.

Run llm-lab in the root directory of the workspace, or set `LLM_LAB_WORKSPACE_DIR`. Your machine needs `git`, `ssh` and `rsync`.

## Lifecycle

1. `llm-lab jobs submit <manifest>` rents the offer with the lowest price that fits the placement and records the job.
2. The job writes its exit code to the instance log. Run `llm-lab jobs status <name>` to record the result and destroy the instance. If you do not, the instance destroys itself after `ttl_seconds_after_finished`.
3. `llm-lab jobs list`, `llm-lab jobs history <name>` and `llm-lab jobs sync` show and refresh the recorded jobs.

## Stopping a job and SSH access

Each job gets a throwaway SSH key pair and a throwaway host key when you submit it. The instance receives the public half of your key and the private half of its host key. Your machine keeps the private key and pins the host key under `.llm-lab/access/<job name>`, and llm-lab deletes them when the job finishes. The container runs `sshd` on port 2222, and Vast.ai maps that port to a public port, so a job rents only offers that have direct ports.

- `llm-lab jobs ssh <name>` opens a shell on the instance of a running job.
- `llm-lab jobs destroy <name>` first downloads the artifacts directory of the job with rsync to `.llm-lab/downloads/<name>`, then destroys the instance. If the download fails, the instance keeps running. Add `--no-download` to destroy the instance at once.

## Setup

1. Run `uv sync` to install the workspace.
2. Run `llm-lab db upgrade` to create the job database.
3. Run `llm-lab secrets set VAST_API_KEY`. Run `llm-lab secrets set HF_TOKEN` if a job lists `HF_TOKEN` in `secret_environment`. Use a fine-grained token, because the instance runs on a machine that you do not control.

Jobs rent only hosts whose driver supports CUDA 13.0 or newer, unless the placement sets another `min_cuda`.

## Base image

The base image holds the parts that change slowly: the CUDA toolkit, vLLM in `/opt/vllm`, uv, one Python version, `sshd` and `rsync`. It holds no llm-lab code and no job code.

The Image workflow builds `docker/base/Dockerfile` and pushes it to `ghcr.io/estasney/llm-lab-base:<commit SHA>` when a commit on `main` changes the Dockerfile. You can also run the workflow from the Actions tab.

To build the image yourself, run this from the repository root:

```sh
docker build -f docker/base/Dockerfile -t <registry>/llm-lab-base:<tag> .
```

The build args:

- `CUDA_IMAGE`: the parent image. It is a CUDA devel image, because vLLM compiles GPU kernels at runtime with `nvcc`.
- `UV_IMAGE`: the image that supplies uv.
- `PYTHON_VERSION`: the Python version, which uv installs.
- `VLLM_VERSION`: the vLLM version.
- `TORCH_BACKEND`: the PyTorch build. It must match the CUDA version of `CUDA_IMAGE`. The vLLM Dockerfile of each version names that CUDA version.

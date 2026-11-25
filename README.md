# llm-lab

llm-lab generates synthetic datasets with [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) and a model that vLLM serves on a rented Vast.ai GPU.

## Flow

1. `llm-lab datasets submit` runs on your machine. It reads the recipe, validates it with Data Designer, and makes sure that an existing dataset repository on the Hugging Face Hub stores a recipe that generates the same rows. Then it submits a job through `llm-lab jobs`.
2. The job runs the datasets image. The image runs `llm-lab datasets create`, which serves the model of the recipe with vLLM, generates the rows with Data Designer, and uploads them to a private Hub repository in one commit.
3. The job writes its exit code to the instance log. Run `llm-lab jobs status <dataset name>` to record the result and destroy the instance. If you do not, the instance destroys itself after `ttl_seconds_after_finished`.

## Stopping a job and SSH access

Each job gets a throwaway SSH key pair and a throwaway host key when you submit it. The instance receives the public half of your key and the private half of its host key. Your machine keeps the private key and pins the host key under `.llm-lab/access/<job name>`, and llm-lab deletes them when the job finishes. The container runs `sshd` on port 2222, and Vast.ai maps that port to a public port, so a job rents only offers that have direct ports.

- `llm-lab jobs ssh <name>` opens a shell on the instance of a running job.
- `llm-lab jobs destroy <name>` first downloads the artifacts directory of the job with rsync to `.llm-lab/downloads/<name>`, then destroys the instance. If the download fails, the instance keeps running. Add `--no-download` to destroy the instance at once.

Data Designer writes rows to disk in groups of 1,000, so a download gets only the finished groups. Your machine needs `ssh` and `rsync`.

## Recipes

A recipe is a YAML file in `recipes/`. Git tracks the recipes. The Hub repository of each dataset also holds a copy of the recipe as `recipe.yaml`.

A recipe has three sections:

- `job`: `disk_gb`, `ttl_seconds_after_finished` and `placement`, with the same meaning as in a job manifest.
- `vllm_args`: optional arguments that llm-lab adds to `vllm serve`, for example `--tensor-parallel-size` or `--max-model-len`. llm-lab sets the model, the host and the port.
- `data_designer`: a Data Designer configuration. All of its model configs must name the same model and the same provider. vLLM serves that model, and llm-lab points that provider to vLLM.

`recipes/recipe.schema.json` is the JSON schema of a recipe. To let your IDE validate a recipe, put `# yaml-language-server: $schema=recipe.schema.json` on the first line of the recipe. After you change the recipe model or upgrade Data Designer, run `llm-lab datasets schema > recipes/recipe.schema.json`. A test fails when the file is out of date.

## Setup

1. Run `llm-lab db upgrade` to create the job database.
2. Run `llm-lab secrets set VAST_API_KEY` and `llm-lab secrets set HF_TOKEN`. The job receives the Hugging Face token to download the model and to upload the dataset. Use a fine-grained token, because the instance runs on a machine that you do not control.
3. Set `LLM_LAB_DATASETS_IMAGE` to `ghcr.io/estasney/llm-lab-datasets:v<version>`, where `<version>` is the llm-lab version that you run.

Jobs rent only hosts whose driver supports CUDA 13.0 or newer, unless the placement sets another `min_cuda`.

## Image

The Version workflow tags each merge to `main` and starts the Image workflow for that tag. The Image workflow builds `docker/datasets/Dockerfile` and pushes it to `ghcr.io/estasney/llm-lab-datasets:<tag>`. To build the image of an existing tag again, run the Image workflow from the Actions tab and select the tag.

To build the image yourself, run this from the repository root:

```sh
docker build -f docker/datasets/Dockerfile -t <registry>/llm-lab-datasets:<tag> .
```

The build args `PYTHON_IMAGE`, `UV_IMAGE` and `VLLM_VERSION` set the base image, the uv image and the vLLM version.

## Commands

```sh
llm-lab datasets submit recipes/<recipe>.yaml --dataset <namespace>/<name> --rows <rows>
llm-lab jobs status <name>
llm-lab datasets create recipes/<recipe>.yaml --dataset <namespace>/<name> --rows <rows>
```

`datasets create` is the command that the job runs. You can also run it on a machine that has vLLM and a GPU.

## Adding rows to a dataset

When the dataset repository exists, `datasets submit` and `datasets create` add the new rows to it. The commands stop when the `data_designer` or `vllm_args` section of the recipe differs from the `recipe.yaml` of the repository. The `job` section can differ.

Each run uploads its parquet files with a run id in their names, for example `data/<run id>-batch_00000.parquet`, so the files of earlier runs stay. The run rewrites `README.md` and `metadata.json` to describe all rows of the repository. A run uploads all of its files in one commit, so a failed run leaves the repository unchanged. The commit fails when another commit reached the repository after the check.

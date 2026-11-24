# llm-lab

llm-lab generates synthetic datasets with [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) and a model that vLLM serves on a rented Vast.ai GPU.

## Flow

1. `llm-lab datasets submit` runs on your machine. It reads the recipe, validates it with Data Designer, and makes sure that the dataset repository does not exist on the Hugging Face Hub. Then it submits a job through `llm-lab jobs`.
2. The job runs the datasets image. The image runs `llm-lab datasets create`, which serves the model of the recipe with vLLM, generates the rows with Data Designer, and uploads the dataset to a private Hub repository.
3. The job writes its exit code to the instance log. Run `llm-lab jobs status <dataset name>` to record the result and destroy the instance. If you do not, the instance destroys itself after `ttl_seconds_after_finished`.

## Recipes

A recipe is a YAML file in `recipes/`. Git tracks the recipes. The Hub repository of each dataset also holds a copy of the recipe as `recipe.yaml`.

A recipe has three sections:

- `job`: `disk_gb`, `ttl_seconds_after_finished` and `placement`, with the same meaning as in a job manifest.
- `vllm_args`: optional arguments that llm-lab adds to `vllm serve`, for example `--tensor-parallel-size` or `--max-model-len`. llm-lab sets the model, the host and the port.
- `data_designer`: a Data Designer configuration. All of its model configs must name the same model and the same provider. vLLM serves that model, and llm-lab points that provider to vLLM.

## Setup

1. Run `llm-lab db upgrade` to create the job database.
2. Run `llm-lab secrets set VAST_API_KEY` and `llm-lab secrets set HF_TOKEN`. The job receives the Hugging Face token to download the model and to upload the dataset. Use a fine-grained token, because the instance runs on a machine that you do not control.
3. Build the image from the repository root and push it to a registry that Vast.ai can pull from:

   ```sh
   docker build -f docker/datasets/Dockerfile -t <registry>/llm-lab-datasets:<tag> .
   docker push <registry>/llm-lab-datasets:<tag>
   ```

4. Set `LLM_LAB_DATASETS_IMAGE` to the reference of that image.

The Vast.ai host must support CUDA 13.0, which the vLLM base image uses.

## Commands

```sh
llm-lab datasets submit recipes/<recipe>.yaml --dataset <namespace>/<name> --rows <rows>
llm-lab jobs status <name>
llm-lab datasets create recipes/<recipe>.yaml --dataset <namespace>/<name> --rows <rows>
```

`datasets create` is the command that the job runs. You can also run it on a machine that has vLLM and a GPU.

#!/usr/bin/env bash
set -euo pipefail

recipe_file=$(mktemp --suffix=.yaml)
base64 --decode <<< "$LLM_LAB_RECIPE" | gunzip > "$recipe_file"
exec llm-lab datasets create "$recipe_file" "$@"

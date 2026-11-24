#!/usr/bin/env bash
set -euo pipefail

initial_password_file=/run/unsloth/studio-initial-password

if [[ -s "$initial_password_file" ]] && ! /usr/local/bin/unsloth-studio-run --stored; then
    IFS= read -r -d '' UNSLOTH_STUDIO_PASSWORD < "$initial_password_file" || true
    export UNSLOTH_STUDIO_PASSWORD
fi

exec "${UNSLOTH_STUDIO_HOME}/bin/unsloth" studio -H 127.0.0.1 -p "${UNSLOTH_STUDIO_PORT}"

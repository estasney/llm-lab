#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PUBLIC_KEY:-}" ]]; then
    echo "ERROR: PUBLIC_KEY is not set. sshd is the only way into this container." >&2
    exit 1
fi
if [[ -z "${UNSLOTH_STUDIO_PASSWORD:-}" ]]; then
    echo "ERROR: UNSLOTH_STUDIO_PASSWORD is not set." >&2
    exit 1
fi

mkdir -p /root/.ssh /run/sshd /run/unsloth /workspace
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A

(umask 077 && printf '%s' "$UNSLOTH_STUDIO_PASSWORD" > /run/unsloth/studio-initial-password)
unset UNSLOTH_STUDIO_PASSWORD

exec supervisord -c /etc/llm-lab/supervisord.conf

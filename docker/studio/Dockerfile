FROM unsloth/unsloth:v0.1.923-beta@sha256:cbe88f91aef98597c54f04f86763f3f9a6dbce625f84dae0a883c381264b605d

RUN /usr/bin/apt-get update \
 && /usr/bin/apt-get install -y --no-install-recommends rsync \
 && rm -rf /var/lib/apt/lists/*

ENV UNSLOTH_SKIP_NOTEBOOK_SYNC=1

COPY supervisord.conf /etc/llm-lab/supervisord.conf
COPY --chmod=755 entrypoint.sh /usr/local/bin/llm-lab-entrypoint
COPY --chmod=755 launch.sh /usr/local/bin/llm-lab-launch
COPY --chmod=755 studio-run.sh /usr/local/bin/llm-lab-studio-run

ENTRYPOINT ["/usr/local/bin/llm-lab-entrypoint"]
CMD ["/usr/local/bin/llm-lab-launch"]

# Containerfile for template-agent (single image for dev and production)
#
# Agent config is baked in at build time. Override at runtime via volume mount:
#   podman run -v ./config:/app/config:Z -p 5002:5002 template-agent
#
# Build: podman build -t template-agent .
# Run:   podman run -p 5002:5002 template-agent

FROM registry.access.redhat.com/ubi9/python-312:latest

WORKDIR /app
USER root

COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r pyproject.toml && \
    mkdir -p /app/.cache /app/config/agent && \
    chown -R 65532:root /app/.cache /app/config

USER 65532

COPY --chown=65532:root deep_agent /app/deep_agent
COPY --chown=65532:root config /app/config
COPY --chown=65532:root aegra.json /app/aegra.json
COPY --chown=65532:root entrypoint.sh /app/entrypoint.sh

ENV PYTHONPATH=/app
ENV AGENT_HOST=0.0.0.0
ENV AGENT_PORT=5002
ENV AEGRA_CONFIG=/app/aegra.json
ENV CONFIG_PATH=/app/config/agent

EXPOSE 5002

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "-m", "deep_agent.aegra.entrypoint"]

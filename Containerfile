# Containerfile for template-agent (single image for dev and production)
#
# Default agent config is baked in at /app/config/agent/.
# Mount config at /app/config (compose) or /app/config/agent (K8s ConfigMap/PVC)
# to override without rebuilding the image.
#
# Build: podman build -t template-agent .
# Run:   podman run -v /path/to/config:/app/config:ro -p 5002:5002 template-agent

ARG PYTHON_TAG=3.14.4-builder
FROM registry.access.redhat.com/hi/python:${PYTHON_TAG}

WORKDIR /app
USER root

COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r pyproject.toml && \
    mkdir -p /app/.cache && chown -R 65532:root /app/.cache

USER 65532

COPY --chown=65532:root deep_agent /app/deep_agent
COPY --chown=65532:root config /app/config
COPY --chown=65532:root aegra.json /app/aegra.json
COPY --chown=65532:root runtime /app/runtime

ENV PYTHONPATH=/app
ENV AGENT_HOST=0.0.0.0
ENV AGENT_PORT=5002
ENV AEGRA_CONFIG=/app/aegra.json
ENV CONFIG_PATH=/app/config/agent

EXPOSE 5002

CMD ["/bin/sh", "-c", "exec /app/.venv/bin/python -m runtime.config_loader"]

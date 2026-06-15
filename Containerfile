# Available tags: latest, 3.14.4, 3.14, 3
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

ENV PYTHONPATH=/app
ENV AGENT_HOST=0.0.0.0
ENV AGENT_PORT=5002

EXPOSE 5002
# Run uvicorn directly as PID 1 so logs go to container stdout
# and SIGTERM triggers graceful shutdown. The aegra serve wrapper
# uses subprocess.run() which swallows child output and signals.
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/python -m uvicorn aegra_api.main:app --host ${AGENT_HOST} --port ${AGENT_PORT}"]

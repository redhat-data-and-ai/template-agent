# Containerfile for template-agent (single image for dev and production)
#
# Agent config is NOT baked in — mount config/agent at /app/config/agent
# (compose: ./config:/app/config:ro; K8s: ConfigMap/PVC).
#
# Build: podman build -t template-agent .
# Run:   podman run -v ./config:/app/config:ro -p 5002:5002 template-agent

# Pin at minor level, not patch. Patch-level Hummingbird tags are frozen and do
# not receive security rebuilds: 3.14.4-builder still ships
# ca-certificates-2025.2.80_v9.0.304-7.2.hum1, which carries the eMudhra roots
# removed by CVE-2026-114406. 3.14-builder has 2026.2.92_v9.0.319-2.hum1.
ARG PYTHON_TAG=3.14-builder
FROM registry.access.redhat.com/hi/python:${PYTHON_TAG}

WORKDIR /app
USER root

COPY pyproject.toml /app/pyproject.toml

# pip is not needed at runtime — the app runs from /app/.venv — and its
# vendored deps (pip/_vendor/vendor.txt) ship a known-vulnerable certifi and
# urllib3 that cannot be upgraded independently. pip 26.2 began publishing
# pip/_vendor/bom.cdx.json, so scanners now report those bundled copies as
# image packages even though nothing imports them. Same reasoning as
# eval-runner/Containerfile, with one difference: that image can drop pip
# outright, while this one creates per-agent venvs at runtime
# (deep_agent/src/infrastructure/backend.py). `python -m venv` seeds those from
# /usr/share/python-wheels, so the wheel must stay — refreshed to pip 26.2.2,
# which vendors the fixed certifi 2026.10.19. Do not remove the wheel swap:
# without it, every runtime agent venv gets a pip carrying the eMudhra roots.
RUN pip install --no-cache-dir uv && \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r pyproject.toml && \
    pip download --no-deps --only-binary=:all: -d /tmp/pipwheel pip==26.2.2 && \
    rm -f /usr/share/python-wheels/pip-*.whl && \
    mv /tmp/pipwheel/pip-*.whl /usr/share/python-wheels/ && \
    rm -rf /usr/lib/python3.14/site-packages/pip \
           /usr/lib/python3.14/site-packages/pip-*.dist-info \
           /usr/sbin/pip /usr/sbin/pip3 /tmp/pipwheel && \
    mkdir -p /app/.cache /app/config/agent && \
    chown -R 65532:root /app/.cache /app/config && \
    chown 65532:0 /app && chmod g+w /app

USER 65532

COPY --chown=65532:root deep_agent /app/deep_agent
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

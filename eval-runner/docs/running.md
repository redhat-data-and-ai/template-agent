# Running the Eval Runner

## Prerequisites

The root stack (Postgres + agent) must be running before starting the eval runner.

```bash
# From the repo root — start agent + Postgres
make local
```

The eval runner expects the agent at `http://127.0.0.1:5002` by default.

---

## Option 1 — Local venv (fastest, recommended for dev)

```bash
cd eval-runner
make local
```

What this does:
1. Creates `.venv` and installs dependencies if not already present (`make venv`).
2. Starts `uvicorn eval_api:app` on port **8099** with the standard local env vars pre-set:

   | Variable | Value set by `make local` |
   |---|---|
   | `AGENT_HOST` | `http://127.0.0.1:5002` |
   | `POSTGRES_HOST` | `localhost` |
   | `POSTGRES_PORT` | `5432` |
   | `POSTGRES_DB` | `template_agent` |
   | `POSTGRES_USER` | `postgres` |
   | `POSTGRES_PASSWORD` | `postgres` |
   | `AGENT_CONFIG_DIR` | `../config/agent` |

Any variables already set in `../.env` (the root `.env`) are sourced first, so
`VLLM_BASE_URL` and `VLLM_API_KEY` carry over automatically if present.

Press `Ctrl+C` to stop. The Makefile cleans up the port on exit.

---

## Option 2 — Container (closer to production)

### Build the image

```bash
cd eval-runner
make build
```

Builds `template-agent-eval-runner:local` using `Containerfile`. Requires `podman`.

### Start as container (joins root stack network)

```bash
make up
```

Equivalent to `podman-compose --profile eval up --build -d`. The container joins
the `agent-network` defined in the root `compose.yaml` so it can reach the agent
and Postgres by service name.

### Other container commands

```bash
make logs     # tail container logs
make down     # stop and remove the container
```

---

## Option 3 — OpenShift

```bash
cd eval-runner
make deploy NAMESPACE=your-project
```

Applies manifests from `deployment/overlays/openshift/` and triggers an
`oc start-build` from the local source tree. Requires `oc` CLI and an active
login to the cluster.

After deploy:
```bash
oc logs -l app=eval-runner --tail=100
oc get pods -l app=eval-runner
```

---

## Verifying the service is up

```bash
curl http://localhost:8099/health
# {"status": "ok"}
```

---

## Running tests

```bash
cd eval-runner
make test
```

Installs test dependencies into the venv, runs pytest with coverage, and fails
if coverage drops below 90%.

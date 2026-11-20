# Kind Cluster Deployment

Deploy the full stack (UI + Agent + MCP Server + infrastructure) to a local Kubernetes cluster using [Kind](https://kind.sigs.k8s.io/).

## Prerequisites

- `kind` — [install](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- `kubectl`
- `podman` or `docker` (for building images)

## Quick Start

```bash
make kind
```

This single command will:

1. Clone `template-mcp-server` and `template-ui` repos into `.kind/`
2. Create a Kind cluster with ingress support
3. Build all three images (agent, MCP server, UI) and load them into Kind
4. Deploy the full stack via Kustomize
5. Wait for all pods to be ready

## What's deployed

| Service | Image | Port | Ingress |
|---------|-------|------|---------|
| UI | template-ui:local | 8080 | http://ui.localhost |
| Agent | template-agent:local | 5002 | http://agent.localhost |
| MCP Server | template-mcp-server:local | 5001 | http://mcp.localhost |
| Postgres (pgvector) | pgvector/pgvector:pg16 | 5432 | — |
| Redis | redis:7-alpine | 6379 | — |
| Jaeger | jaegertracing/all-in-one | 16686 | http://jaeger.localhost |

## Useful Commands

```bash
kubectl -n template-agent get pods
kubectl -n template-agent logs -l component=agent -f
kubectl -n template-agent logs -l component=mcp-server -f
kubectl -n template-agent logs -l component=ui -f
```

## Port-Forward (alternative to Ingress)

```bash
kubectl -n template-agent port-forward svc/ui 8080:8080
kubectl -n template-agent port-forward svc/agent 5002:5002
kubectl -n template-agent port-forward svc/mcp-server 5001:5001
```

## Secrets (Langfuse, Daytona)

`deployment/base/secret.yaml` ships every credential key as an empty
string and is **tracked in git** — never put a live key in it. Supply
real values to a running cluster instead, so nothing secret is ever
committed:

```bash
kubectl -n template-agent patch secret agent-secrets --type merge -p "$(cat <<EOF
{"stringData": {
  "LANGFUSE_PUBLIC_KEY": "$LANGFUSE_PUBLIC_KEY",
  "LANGFUSE_SECRET_KEY": "$LANGFUSE_SECRET_KEY",
  "LANGFUSE_BASE_URL":   "$LANGFUSE_BASE_URL",
  "DAYTONA_API_KEY":     "$DAYTONA_API_KEY"
}}
EOF
)"
kubectl -n template-agent rollout restart deployment/agent
```

Export those four from your local `.env` first (`set -a; . .env; set +a`),
which is gitignored and is where `make kind` already expects them.

Both integrations are optional and fail soft. With no Langfuse keys the
agent simply does not trace; with no `DAYTONA_API_KEY` the sandbox tools
report that they are not configured and the agent still answers.

Traces are separated by Langfuse **project**, one per deployment, not by
environment tag: each namespace's `agent-secrets` carries the keypair for
its own project, so `ta-daytona` and `ta-openshell` report to different
projects and never mix. Creating the project and its keypair is a
Langfuse-side step — project API keys cannot create projects, so a new
preview needs its keypair made in the Langfuse UI (or with an
organization-scoped key) before it can trace.

`LANGFUSE_TRACING_ENVIRONMENT` is not a secret and is a *secondary* axis
— it separates stages within one project (local vs deployed), not one
deployment from another.

## Differences from OpenShift

| Concern | Kind | OpenShift |
|---------|------|-----------|
| Image build | Local `podman build` + `kind load` | BuildConfig (in-cluster) |
| Routing | NGINX Ingress | Route |
| Image pull | `imagePullPolicy: Never` | ImageStream |
| Security | Default PSA | SCC (restricted) |
| Storage | Default StorageClass | OpenShift PVs |

## Teardown

```bash
make kind-down
```

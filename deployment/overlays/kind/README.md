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

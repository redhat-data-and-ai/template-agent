# OPA Kustomize Component Design

**Date**: 2026-07-22
**Branch**: feature/add-opa-support
**Scope**: `deployment/components/opa/` — production Kustomize manifests for the OPA authorization service

## Problem

PR #121 added OPA to `compose.yaml` for local development, but no production deployment plumbing exists. There is no `deployment/components/opa/` directory and no Kustomize manifests (Deployment, Service, ConfigMaps, Secret) for OPA.

## Approach

Follow the existing `kind: Component` pattern used by `deployment/components/postgres` and `deployment/components/redis`. Create a self-contained optional Kustomize component that overlays activate by adding `- ../../components/opa` to their `components:` block.

## File Structure

```
deployment/components/opa/
├── kustomization.yaml       # kind: Component; patches agent-config with OPA_URL
├── deployment.yaml          # OPA Deployment — image placeholder "opa"
├── service.yaml             # ClusterIP, port 8181, selector: component=opa
├── policies-configmap.yaml  # ConfigMap "opa-policies" — empty scaffold for Rego files
├── configmap.yaml           # ConfigMap "opa-config" — git URL, branch, poll interval, SSL
└── secret.yaml              # Secret "opa-secrets" — git auth user + token (empty defaults)
```

## Resource Specs

### `kustomization.yaml`

- `apiVersion: kustomize.config.k8s.io/v1alpha1`, `kind: Component`
- Resources: all 5 files above
- Patch: JSON6902 add on `ConfigMap/agent-config` to inject `OPA_URL: "http://opa:8181/v1/data/agent/authz"`

### `service.yaml`

- Name: `opa`
- Type: `ClusterIP`
- Port: 8181 → targetPort 8181
- Selector: `component: opa`

### `policies-configmap.yaml`

- Name: `opa-policies`
- `data: {}` — empty by default
- Operators add Rego file content as keys (e.g., `authz.rego: |`) per overlay
- Mounted read-only at `/policies` in the OPA container; the reload-watch script loads all `.rego` files from this directory as the bundled baseline alongside git-fetched policies

### `configmap.yaml`

- Name: `opa-config`
- Keys (all non-sensitive, safe to patch in overlay configmap patches):

| Key | Default | Notes |
|-----|---------|-------|
| `OPA_POLICY_GIT_REPO` | `""` | Empty — must be set per overlay to enable git fetching |
| `OPA_POLICY_GIT_BRANCH` | `"main"` | |
| `OPA_POLICY_GIT_SUBDIR` | `""` | Sparse checkout subdir |
| `OPA_POLICY_GIT_SSL_VERIFY` | `"true"` | Set `"false"` for self-signed certs |
| `OPA_POLL_INTERVAL` | `"30"` | Seconds; compose default is 2 (too aggressive for prod) |

### `secret.yaml`

- Name: `opa-secrets`
- Type: `Opaque`
- Keys: `OPA_POLICY_GIT_AUTH_USER: ""`, `OPA_POLICY_GIT_AUTH_TOKEN: ""`
- Empty defaults; overlays patch with real credentials

### `deployment.yaml`

- Name: `opa`
- Labels: `component: opa`
- Replicas: 1 (stateless policy evaluator — no need to scale horizontally)
- Image: `opa` (no tag) — overridden per overlay via `images:` block
- Port: 8181

**Volume mounts:**

| Volume | Type | Mount path | Mode |
|--------|------|-----------|------|
| `opa-policies` ConfigMap | ConfigMap | `/policies` | ReadOnly |

**Env vars:**

- All keys from `opa-config` ConfigMap via `envFrom.configMapRef`
- All keys from `opa-secrets` Secret via `envFrom.secretRef`

**Probes** (matches compose healthcheck):

```yaml
livenessProbe:
  exec:
    command: ["/usr/local/bin/opa", "eval", "true"]
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3
readinessProbe:
  exec:
    command: ["/usr/local/bin/opa", "eval", "true"]
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

**Security context** (matches user created in `opa/Containerfile`):

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop: [ALL]
```

**Resources** (lightweight — OPA is a policy evaluator, not a data store):

```yaml
resources:
  requests:
    memory: "64Mi"
    cpu: "50m"
  limits:
    memory: "128Mi"
    cpu: "200m"
```

## Overlay Integration

To enable OPA in an overlay, add to its `kustomization.yaml`:

```yaml
components:
  - ../../components/opa
```

To configure git-based policy fetching, add a patch targeting `ConfigMap/opa-config` and a patch targeting `Secret/opa-secrets`. To supply bundled baseline policies, add a patch targeting `ConfigMap/opa-policies` with Rego file content as keys.

The `openshift` and `openshift-headless` overlays may also want resource limit patches on the OPA Deployment (following the same pattern used for postgres and redis in those overlays).

## What Is Not In Scope

- Changes to existing overlays — activating OPA per overlay is left to the operator
- Rego policy content — `opa-policies` is intentionally empty; policy authoring is out of scope
- HPA/PDB for OPA — the service is stateless and single-replica; scaling policy is left to operators

# OPA Kustomize Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `deployment/components/opa/` — a self-contained Kustomize Component that deploys OPA and configures the agent to connect to it.

**Architecture:** Six YAML files following the exact same `kind: Component` pattern as `deployment/components/postgres` and `deployment/components/redis`. The component creates an OPA Deployment, ClusterIP Service, a policies ConfigMap (empty scaffold), a non-sensitive config ConfigMap, and a credentials Secret. It also patches `agent-config` to inject `OPA_URL` so the agent knows where OPA is. Overlays opt in by adding `- ../../components/opa` to their `components:` block.

**Tech Stack:** Kustomize (v1alpha1 Component API), Kubernetes manifests, `kubectl kustomize` for validation.

## Global Constraints

- All files live under `deployment/components/opa/`
- `kustomization.yaml` must use `apiVersion: kustomize.config.k8s.io/v1alpha1` and `kind: Component` — same as postgres and redis
- OPA container image name: `opa` (no registry, no tag) — overlays override via `images:` block
- OPA user in container: uid 1000 (matches `opa/Containerfile` `adduser -u 1000`)
- Port: 8181
- All env vars sourced via individual `env:` entries (not `envFrom:`) — matches existing postgres/redis pattern
- No commits — user has requested no git commits during this session

---

### Task 1: Create service.yaml and policies-configmap.yaml

**Files:**
- Create: `deployment/components/opa/service.yaml`
- Create: `deployment/components/opa/policies-configmap.yaml`

**Interfaces:**
- Produces: `Service/opa` on port 8181 (consumed by the agent via `OPA_URL: http://opa:8181/...`)
- Produces: `ConfigMap/opa-policies` mounted at `/policies` (consumed by Task 3's Deployment)

- [ ] **Step 1: Create service.yaml**

```yaml
# deployment/components/opa/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: opa
  labels:
    component: opa
spec:
  type: ClusterIP
  ports:
    - port: 8181
      targetPort: 8181
      protocol: TCP
      name: http
  selector:
    component: opa
```

- [ ] **Step 2: Create policies-configmap.yaml**

```yaml
# deployment/components/opa/policies-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opa-policies
data: {}
```

`data: {}` is intentional — this is an empty scaffold. Kubernetes mounts an empty ConfigMap as an empty directory. OPA starts fine with no baseline Rego files; git fetching (if configured) provides policies at runtime. Operators populate this per overlay by patching with Rego file content as keys.

---

### Task 2: Create secret.yaml and configmap.yaml

**Files:**
- Create: `deployment/components/opa/secret.yaml`
- Create: `deployment/components/opa/configmap.yaml`

**Interfaces:**
- Produces: `Secret/opa-secrets` with keys `OPA_POLICY_GIT_AUTH_USER` and `OPA_POLICY_GIT_AUTH_TOKEN` (consumed by Task 3's Deployment)
- Produces: `ConfigMap/opa-config` with git and polling settings (consumed by Task 3's Deployment)

- [ ] **Step 1: Create secret.yaml**

```yaml
# deployment/components/opa/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: opa-secrets
type: Opaque
stringData:
  OPA_POLICY_GIT_AUTH_USER: ""
  OPA_POLICY_GIT_AUTH_TOKEN: ""
```

Empty defaults — overlays patch with real credentials for private repos.

- [ ] **Step 2: Create configmap.yaml**

```yaml
# deployment/components/opa/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opa-config
data:
  OPA_POLICY_GIT_REPO: ""
  OPA_POLICY_GIT_BRANCH: "main"
  OPA_POLICY_GIT_SUBDIR: ""
  OPA_POLICY_GIT_SSL_VERIFY: "true"
  OPA_POLL_INTERVAL: "30"
```

`OPA_POLICY_GIT_REPO` is empty by default — when empty, the `opa-reload-watch.sh` script skips git fetching and serves only the bundled policies from `/policies`. Set it per overlay to enable git-based policy loading. `OPA_POLL_INTERVAL` is 30s (not 2s as in compose) to avoid aggressive polling in production.

---

### Task 3: Create deployment.yaml

**Files:**
- Create: `deployment/components/opa/deployment.yaml`

**Interfaces:**
- Consumes: `ConfigMap/opa-config` (env vars for git settings), `Secret/opa-secrets` (git credentials), `ConfigMap/opa-policies` (mounted at `/policies`)
- Produces: `Deployment/opa` — OPA pod running on port 8181

- [ ] **Step 1: Create deployment.yaml**

```yaml
# deployment/components/opa/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opa
  labels:
    component: opa
spec:
  replicas: 1
  selector:
    matchLabels:
      component: opa
  template:
    metadata:
      labels:
        component: opa
    spec:
      containers:
        - name: opa
          image: opa
          ports:
            - containerPort: 8181
              name: http
              protocol: TCP
          env:
            - name: OPA_POLICY_GIT_REPO
              valueFrom:
                configMapKeyRef:
                  name: opa-config
                  key: OPA_POLICY_GIT_REPO
            - name: OPA_POLICY_GIT_BRANCH
              valueFrom:
                configMapKeyRef:
                  name: opa-config
                  key: OPA_POLICY_GIT_BRANCH
            - name: OPA_POLICY_GIT_SUBDIR
              valueFrom:
                configMapKeyRef:
                  name: opa-config
                  key: OPA_POLICY_GIT_SUBDIR
            - name: OPA_POLICY_GIT_SSL_VERIFY
              valueFrom:
                configMapKeyRef:
                  name: opa-config
                  key: OPA_POLICY_GIT_SSL_VERIFY
            - name: OPA_POLL_INTERVAL
              valueFrom:
                configMapKeyRef:
                  name: opa-config
                  key: OPA_POLL_INTERVAL
            - name: OPA_POLICY_GIT_AUTH_USER
              valueFrom:
                secretKeyRef:
                  name: opa-secrets
                  key: OPA_POLICY_GIT_AUTH_USER
            - name: OPA_POLICY_GIT_AUTH_TOKEN
              valueFrom:
                secretKeyRef:
                  name: opa-secrets
                  key: OPA_POLICY_GIT_AUTH_TOKEN
          livenessProbe:
            exec:
              command:
                - /usr/local/bin/opa
                - eval
                - "true"
            initialDelaySeconds: 15
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          readinessProbe:
            exec:
              command:
                - /usr/local/bin/opa
                - eval
                - "true"
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "200m"
          volumeMounts:
            - name: opa-policies
              mountPath: /policies
              readOnly: true
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
      volumes:
        - name: opa-policies
          configMap:
            name: opa-policies
      restartPolicy: Always
```

Notes:
- `image: opa` with no tag — overlays must add an `images:` entry to point to the real registry image built from `opa/Containerfile`
- `runAsUser: 1000` matches the `opa` user created in `opa/Containerfile` (`adduser -u 1000`)
- Liveness and readiness both use `opa eval true` — same command as the compose healthcheck
- `readOnly: true` on the `/policies` mount is safe because `opa-reload-watch.sh` only reads from that directory; git clones go to `/tmp/policy-repo`

---

### Task 4: Create kustomization.yaml and validate

**Files:**
- Create: `deployment/components/opa/kustomization.yaml`

**Interfaces:**
- Consumes: all 5 files created in Tasks 1–3
- Produces: a valid Kustomize Component that any overlay can activate with `- ../../components/opa`

- [ ] **Step 1: Create kustomization.yaml**

```yaml
# deployment/components/opa/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1alpha1
kind: Component

resources:
  - deployment.yaml
  - service.yaml
  - policies-configmap.yaml
  - configmap.yaml
  - secret.yaml

patches:
  - target:
      kind: ConfigMap
      name: agent-config
    patch: |-
      - op: add
        path: /data/OPA_URL
        value: "http://opa:8181/v1/data/agent/authz"
```

- [ ] **Step 2: Validate the component builds without errors**

Run from the repo root:

```bash
kubectl kustomize deployment/components/opa/
```

Expected: No errors. Output should include 5 resources: `Deployment/opa`, `Service/opa`, `ConfigMap/opa-policies`, `ConfigMap/opa-config`, `Secret/opa-secrets`.

Note: A lone Component cannot be built directly by `kubectl kustomize` — it requires a parent Kustomization. If you see `"Component" is not a valid Kustomization` use the next step instead.

- [ ] **Step 3: Validate the component integrates with the base**

Create a temporary test kustomization:

```bash
REPO=/Users/prchaudh/Desktop/Projects/prithviraj-chaudhuri/template-agent
mkdir -p /tmp/test-opa-component
cat > /tmp/test-opa-component/kustomization.yaml << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ${REPO}/deployment/base

components:
  - ${REPO}/deployment/components/opa
EOF
```

Then run:

```bash
kubectl kustomize /tmp/test-opa-component/
```

Expected output includes:
- `kind: ConfigMap` with `name: agent-config` containing `OPA_URL: http://opa:8181/v1/data/agent/authz`
- `kind: ConfigMap` with `name: opa-policies`
- `kind: ConfigMap` with `name: opa-config`
- `kind: Secret` with `name: opa-secrets`
- `kind: Service` with `name: opa`
- `kind: Deployment` with `name: opa`

Verify the patch applied by grepping the output:

```bash
kubectl kustomize /tmp/test-opa-component/ | grep -A1 "OPA_URL"
```

Expected:
```
OPA_URL: http://opa:8181/v1/data/agent/authz
```

- [ ] **Step 4: Clean up the temp directory**

```bash
rm -rf /tmp/test-opa-component
```

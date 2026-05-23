#!/usr/bin/env bash
# Sync deployment manifests from config/deployment/values.yaml.
#
# Reads the single source of truth (config/deployment/values.yaml) and
# updates the OpenShift kustomization, configmap, and secret files to match.
#
# Requires: yq v4+ (https://github.com/mikefarah/yq)
#
# Usage:
#     ./scripts/sync-deployment-config.sh
#     ./scripts/sync-deployment-config.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALUES="$REPO_ROOT/config/deployment/values.yaml"
DEPLOY_DIR="$REPO_ROOT/deployment/openshift"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) echo "Usage: ${0##*/} [--dry-run]"; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

if ! command -v yq &>/dev/null; then
    echo "ERROR: yq v4+ is required (https://github.com/mikefarah/yq)" >&2
    exit 1
fi

if [[ ! -f "$VALUES" ]]; then
    echo "ERROR: $VALUES not found" >&2
    exit 1
fi

echo "Source: config/deployment/values.yaml"
$DRY_RUN && echo "Mode: dry-run" || echo "Mode: write"
echo

write_output() {
    local relpath="$1" content="$2"
    if $DRY_RUN; then
        printf -- '--- %s ---\n%s\n' "$relpath" "$content"
    else
        printf '%s\n' "$content" > "$REPO_ROOT/$relpath"
        echo "  Updated $relpath"
    fi
}

# ── ConfigMap ────────────────────────────────────────────────────────
configmap=$(yq eval '
    {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": (.app.name + "-config"),
            "labels": {"app": .app.name, "component": .app.component}
        },
        "data": (.config | with_entries(.value |= tostring))
    }
' "$VALUES")
write_output "deployment/openshift/configmap.yaml" "$configmap"

# ── Secret ───────────────────────────────────────────────────────────
secret=$(yq eval '
    {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": (.app.name + "-secrets"),
            "labels": {"app": .app.name, "component": .app.component}
        },
        "type": "Opaque",
        "stringData": (.secrets | with_entries(.value |= tostring))
    }
' "$VALUES")
write_output "deployment/openshift/secret.yaml" "$secret"

# ── Kustomization ────────────────────────────────────────────────────
# Build JSON-Patch arrays as YAML strings; strenv() injects them as
# literal block scalars into the final kustomization document.
export DEPLOY_PATCH
DEPLOY_PATCH=$(yq eval '
    [
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/memory",        "value": .resources.requests.memory},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/cpu",           "value": .resources.requests.cpu},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory",          "value": .resources.limits.memory},
        {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu",             "value": .resources.limits.cpu},
        {"op": "replace", "path": "/spec/template/spec/containers/0/ports/0/containerPort",            "value": .container.port},
        {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/initialDelaySeconds",  "value": .probes.liveness.initialDelaySeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/periodSeconds",        "value": .probes.liveness.periodSeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/timeoutSeconds",       "value": .probes.liveness.timeoutSeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/failureThreshold",     "value": .probes.liveness.failureThreshold},
        {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/initialDelaySeconds", "value": .probes.readiness.initialDelaySeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/periodSeconds",       "value": .probes.readiness.periodSeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/timeoutSeconds",      "value": .probes.readiness.timeoutSeconds},
        {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/failureThreshold",    "value": .probes.readiness.failureThreshold}
    ]
' "$VALUES")

export BUILD_PATCH
BUILD_PATCH=$(yq eval '
    [
        {"op": "replace", "path": "/spec/resources/requests/memory", "value": .build.resources.requests.memory},
        {"op": "replace", "path": "/spec/resources/requests/cpu",    "value": .build.resources.requests.cpu},
        {"op": "replace", "path": "/spec/resources/limits/memory",   "value": .build.resources.limits.memory},
        {"op": "replace", "path": "/spec/resources/limits/cpu",      "value": .build.resources.limits.cpu}
    ]
' "$VALUES")

kustomization=$(yq eval '
    {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": [
            "buildconfig.yaml",
            "imagestream.yaml",
            "configmap.yaml",
            "secret.yaml",
            "deployment.yaml",
            "service.yaml",
            "route.yaml"
        ],
        "labels": [
            {
                "pairs": {"app": .app.name, "component": .app.component},
                "includeSelectors": true
            }
        ],
        "replicas": [{"name": .app.name, "count": .app.replicas}],
        "images":   [{"name": .image.name, "newTag": .image.tag}],
        "patches": [
            {
                "target": {"kind": "Deployment", "name": .app.name},
                "patch": strenv(DEPLOY_PATCH)
            },
            {
                "target": {"kind": "BuildConfig", "name": .app.name},
                "patch": strenv(BUILD_PATCH)
            }
        ]
    }
' "$VALUES")
write_output "deployment/openshift/kustomization.yaml" "$kustomization"

$DRY_RUN || echo -e "\nDone. Review changes with: git diff deployment/openshift/"

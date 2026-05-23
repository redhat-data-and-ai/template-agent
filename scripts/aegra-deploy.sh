#!/bin/bash
## Aegra deployment script for Kubernetes/OpenShift (MR-37).
##
## Usage:
##   ./scripts/aegra-deploy.sh [build|deploy|status|teardown]
##
## Requires: oc or kubectl, Docker or Podman

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${AEGRA_IMAGE:-template-agent-aegra}"
NAMESPACE="${AEGRA_NAMESPACE:-$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null || echo default)}"

# Prefer oc if available, fall back to kubectl
KUBECTL=$(command -v oc 2>/dev/null || command -v kubectl 2>/dev/null || echo "")
if [ -z "$KUBECTL" ]; then
    echo "Error: neither oc nor kubectl found in PATH"
    exit 1
fi

log() { echo "[aegra] $*"; }

cmd_build() {
    log "Building aegra Docker image: $IMAGE_NAME"
    cd "$PROJECT_ROOT"

    if command -v langgraph &>/dev/null; then
        langgraph build -t "$IMAGE_NAME"
    else
        log "langgraph CLI not found — using docker build"
        docker build -f Containerfile -t "$IMAGE_NAME" .
    fi

    log "Image built: $IMAGE_NAME"
}

cmd_deploy() {
    log "Deploying to namespace: $NAMESPACE"
    cd "$PROJECT_ROOT"

    $KUBECTL apply -k deployment/aegra/ -n "$NAMESPACE"

    log "Waiting for rollout..."
    $KUBECTL rollout status deployment/aegra-agent -n "$NAMESPACE" --timeout=120s

    log "Deployment complete!"
    cmd_status
}

cmd_status() {
    log "Status (namespace: $NAMESPACE):"
    echo ""
    $KUBECTL get pods -l app=aegra-agent -n "$NAMESPACE" -o wide
    echo ""
    $KUBECTL get svc -l app=aegra-agent -n "$NAMESPACE"
    echo ""
    log "Useful commands:"
    log "  Logs:   $KUBECTL logs -l app=aegra-agent -n $NAMESPACE --tail=100"
    log "  Shell:  $KUBECTL exec -it deploy/aegra-agent -n $NAMESPACE -- bash"
    log "  Port:   $KUBECTL port-forward svc/aegra-agent -n $NAMESPACE 2024:2024"
}

cmd_teardown() {
    log "Tearing down aegra deployment in namespace: $NAMESPACE"
    $KUBECTL delete -k deployment/aegra/ -n "$NAMESPACE" --ignore-not-found
    log "Teardown complete"
}

case "${1:-help}" in
    build)    cmd_build ;;
    deploy)   cmd_deploy ;;
    status)   cmd_status ;;
    teardown) cmd_teardown ;;
    *)
        echo "Usage: $0 [build|deploy|status|teardown]"
        echo ""
        echo "Commands:"
        echo "  build     Build the aegra Docker image"
        echo "  deploy    Deploy to Kubernetes/OpenShift"
        echo "  status    Show deployment status"
        echo "  teardown  Remove aegra deployment"
        exit 1
        ;;
esac

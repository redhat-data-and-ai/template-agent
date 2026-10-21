.PHONY: local dev test clean deploy undeploy kind kind-down container container-down local-down test-triggers test-integration

# OpenShift namespace (can be overridden: make deploy openshift NAMESPACE=my-project)
NAMESPACE ?= $(shell oc project -q 2>/dev/null)

# Dependency checks
deps:
	@which uv > /dev/null && echo "uv: $(shell uv --version)" || (echo "Error: uv not found. Please install uv." && exit 1)
	@which podman > /dev/null && echo "podman: $(shell podman --version)" || (echo "Error: podman not found. Please install podman." && exit 1)
	@which podman-compose > /dev/null && echo "podman-compose: $(shell podman-compose --version)" || (echo "Error: podman-compose not found. Please install podman-compose." && exit 1)
	@which oc > /dev/null && echo "oc: $(shell oc version --client)" || (echo "Error: oc not found. Please install oc." && exit 1)

# Install Python dependencies
install:
	@echo "Creating virtual environment..."
	@test -d .venv || uv venv
	@echo "Installing package with dev dependencies..."
	@. .venv/bin/activate && uv pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	@. .venv/bin/activate && pre-commit install
	@echo "Python dependencies installed successfully!"
	@echo "Activating virtual environment..."
	@echo '#!/bin/bash' > /tmp/activate_and_shell.sh
	@echo 'source .venv/bin/activate' >> /tmp/activate_and_shell.sh
	@echo 'echo "Virtual environment activated! Type exit to return to your original shell."' >> /tmp/activate_and_shell.sh
	@echo 'exec "$$SHELL"' >> /tmp/activate_and_shell.sh
	@chmod +x /tmp/activate_and_shell.sh
	@exec /tmp/activate_and_shell.sh

clean: ## Remove build artifacts, venv, and tear down compose stack
	@echo "Stopping agent on port 5002 (if running)..."
	@lsof -ti :5002 | xargs kill -9 2>/dev/null || true
	@echo "Stopping compose stack (if running)..."
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container down -v 2>/dev/null || true
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml stop pgvector redis opa 2>/dev/null || true
	@podman rmi template-agent_template-agent 2>/dev/null || true
	@echo "Cleaning up build artifacts..."
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -f Current-IT-Root-CAs.pem
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "Cleanup complete"

test:
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	.venv/bin/python -m pytest tests/unit

test-cov: ## Run unit tests with coverage report
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	@echo "Running unit tests with coverage..."
	.venv/bin/python -m pytest tests/unit --cov=deep_agent --cov-report=xml --cov-report=html --cov-report=term-missing

test-all:
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	@echo "Running all tests (unit + skills evals)..."
	.venv/bin/python -m pytest

test-skills:
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	@echo "Running skills evaluations..."
	.venv/bin/python -m pytest tests/skills -m skills -v

eval-promptfoo:
	@echo "Running Promptfoo agent evaluations..."
	@echo "Make sure agent is running at http://localhost:5002"
	@cd config/agent/evals/promptfoo && npx promptfoo@latest eval

mock-mcp:
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Starting Mock MCP Server on http://localhost:5001 (Ctrl+C to stop)..."
	@.venv/bin/python tests/mocks/mock_mcp_server.py

local-with-mock:
	@echo "Run in separate terminals:"
	@echo "  Terminal 1: make mock-mcp"
	@echo "  Terminal 2: make local"

local:
	@echo "Setting up local environment..."
	@test -f .env 2>/dev/null || (echo "Creating .env from .env.example..." && cp .env.example .env 2>/dev/null) || true
	@lsof -ti :5002 | xargs kill -9 2>/dev/null || true
	@echo "Cleaning up stale containers from previous naming scheme..."
	@podman rm -f demo-pgvector demo-redis 2>/dev/null || true
	@echo "Starting infrastructure (Postgres + Redis + OPA)..."
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml up -d pgvector redis opa
	@echo "Waiting for Postgres to be ready..."
	@until podman exec template-agent-pgvector pg_isready -U postgres -q 2>/dev/null; do sleep 1; done
	@podman exec template-agent-pgvector psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='aegra'" | grep -q 1 \
		|| podman exec template-agent-pgvector psql -U postgres -c "CREATE DATABASE aegra;"
	@echo "Starting agent with LangGraph Platform..."
	@echo "API available at: http://localhost:5002"
	@echo "Press Ctrl+C to stop the server (Postgres/Redis keep running — use 'make local-down' to stop them)"
	@trap 'lsof -ti :5002 | xargs kill -INT 2>/dev/null || true; sleep 2; lsof -ti :5002 | xargs kill -9 2>/dev/null || true; exit 130' INT TERM; \
	set -a; [ -f .env ] && . .env 2>/dev/null; set +a; \
	REDIS_BROKER_ENABLED=true \
		POSTGRES_HOST=localhost \
		POSTGRES_PORT=5432 \
		POSTGRES_DB=template_agent \
		POSTGRES_USER=postgres \
		POSTGRES_PASSWORD=postgres \
		REDIS_URL=redis://localhost:6379/0 \
		.venv/bin/aegra dev --port 5002 --no-db-check

local-down:
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml stop pgvector redis opa

container:
	@test -f .env || (echo "Creating .env from .env.example..." && cp .env.example .env)
	@echo "Starting stack: pgvector, redis, template-agent, jaeger"
	@echo "Agent:  http://localhost:5002"
	@echo "Jaeger: http://localhost:16686"
	@export PODMAN_COMPOSE_SILENT=true; \
	trap 'export PODMAN_COMPOSE_SILENT=true; podman-compose -f compose.yaml --profile observability down --timeout 10 2>/dev/null || true; exit 130' INT TERM; \
	ENABLE_OTEL=true \
	OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
	ENABLE_OTEL_TRACES=true \
	OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4317 \
	podman-compose --profile observability --no-ansi up --build --force-recreate --remove-orphans --timeout=60

container-down:
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile observability down

# ---------------------------------------------------------------------------
# Development environment targets
# ---------------------------------------------------------------------------

dev: ## Start agent + deps in containers (detached, tail logs)
	@echo "Starting agent stack (pgvector, redis, template-agent)..."
	@echo "Agent: http://localhost:5002"
	@echo ""
	@test -f .env || (echo "Creating .env from .env.example..." && cp .env.example .env)
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container up --build -d
	@echo ""
	@echo "Tailing agent logs (Ctrl+C to stop)..."
	@echo ""
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container logs -f template-agent

dev-down: ## Stop dev stack
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container down

dev-clean: ## Stop dev stack and remove all data
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container down -v
	@echo "All dev data volumes removed"

dev-logs: ## Tail all service logs
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container logs -f

dev-restart: ## Restart dev stack
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container restart

dev-agent: ## Restart just the agent service
	@export PODMAN_COMPOSE_SILENT=true && podman-compose -f compose.yaml --profile container restart template-agent

# Deployment targets
deploy:
	@if [ "$(filter openshift,$(MAKECMDGOALS))" != "openshift" ] && [ "$(filter mpp,$(MAKECMDGOALS))" != "mpp" ]; then \
		echo "Usage: make deploy [openshift|mpp]"; \
		echo "Available deployment targets: openshift, mpp"; \
		exit 1; \
	fi

openshift:
	@echo "Checking for oc CLI..."
	@which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1)
	@echo "Validating namespace..."
	@if [ -z "$(NAMESPACE)" ]; then \
		echo "Error: NAMESPACE not set. Usage: make deploy openshift NAMESPACE=your-project"; \
		exit 1; \
	fi; \
	echo "Using namespace: $(NAMESPACE)"; \
	echo "Switching to namespace..."; \
	oc project $(NAMESPACE) || (echo "Error: Cannot switch to namespace '$(NAMESPACE)'. Check permissions." && exit 1); \
	echo "Updating namespace references..."; \
	sed -i.bak "s|NAMESPACE_PLACEHOLDER|$(NAMESPACE)|g" deployment/overlays/openshift/kustomization.yaml; \
	echo "Creating BuildConfig and ImageStream..."; \
	oc apply -f deployment/overlays/openshift/buildconfig.yaml; \
	oc apply -f deployment/overlays/openshift/imagestream.yaml; \
	oc apply -f deployment/overlays/openshift/eval-runner-buildconfig.yaml; \
	oc apply -f deployment/overlays/openshift/eval-runner-imagestream.yaml; \
	echo "Building agent container image from source..."; \
	oc start-build agent --from-dir=. \
		--exclude='(^|/)\.venv(/|$$)' \
		--exclude='(^|/)__pycache__(/|$$)' \
		--exclude='(^|/)\.pytest_cache(/|$$)' \
		--exclude='(^|/)tests(/|$$)' \
		--exclude='(^|/)examples(/|$$)' \
		--exclude='(^|/)\.mypy_cache(/|$$)' \
		--exclude='(^|/)\.ruff_cache(/|$$)' \
		--exclude='.*\.log$$' \
		--follow || (mv deployment/overlays/openshift/kustomization.yaml.bak deployment/overlays/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	echo "Building eval-runner container image from source..."; \
	oc start-build eval-runner --from-dir=eval-runner --follow || \
		(mv deployment/overlays/openshift/kustomization.yaml.bak deployment/overlays/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	echo "Deploying resources to OpenShift..."; \
	oc apply -k deployment/overlays/openshift/ || (mv deployment/overlays/openshift/kustomization.yaml.bak deployment/overlays/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	rm -f deployment/overlays/openshift/kustomization.yaml.bak; \
	echo "Deployment complete!"; \
	echo "Checking deployment status..."; \
	oc get pods -l app=agent; \
	oc get pods -l app=eval-runner; \
	echo ""; \
	echo "Useful commands:"; \
	echo "  View logs: oc logs -l app=agent --tail=100"; \
	echo "  View eval logs: oc logs -l app=eval-runner --tail=100"; \
	echo "  Get route: oc get route agent"; \
	echo "  Check status: oc get pods,svc,route -l app=agent"

mpp:
	@echo "Error: MPP deployment is not yet implemented."
	@echo "The deployment/mpp/ kustomize overlay has not been created."
	@echo "Use 'make deploy openshift' for OpenShift or 'make kind' for local Kubernetes."
	@exit 1

# ---------------------------------------------------------------------------
# Kind cluster: local Kubernetes testing
# ---------------------------------------------------------------------------

KIND_CLUSTER := template-agent
KIND_CTX := kind-$(KIND_CLUSTER)
KIND_IMAGE := template-agent:local
# Only for clusters with no config volume. Kind mounts one now, so
# `make kind` does not build this — Containerfile.config is kept for
# anywhere that cannot provide a volume.
KIND_AGENT_CONFIG_IMAGE := template-agent-config:local
KIND_MCP_IMAGE := template-mcp-server:local
KIND_UI_IMAGE := template-ui:local
KIND_NS := template-agent
KIND_DIR := .kind
# Rootless podman cannot bind ports below 1024, so the ingress is published on
# high ports. Everything the tests use goes through port-forward regardless.
KIND_HTTP_PORT ?= 8081
KIND_HTTPS_PORT ?= 8444
KIND_MCP_REPO := https://github.com/redhat-data-and-ai/template-mcp-server.git
KIND_MCP_BRANCH ?= main
KIND_UI_REPO := https://github.com/redhat-data-and-ai/template-ui.git
KIND_UI_BRANCH ?= main

# Build the UI from a sibling checkout when there is one, so `make
# kind` deploys the tree you are editing rather than whatever is on
# the remote's default branch. Falls back to the clone for anyone
# working on the agent alone. Override with KIND_UI_DIR=...
KIND_UI_DIR ?= $(if $(wildcard ../template-ui/Containerfile),../template-ui,$(KIND_DIR)/template-ui)
KCTL := kubectl --context $(KIND_CTX)

kind: ## Deploy full stack (agent + MCP + UI) to a local Kind cluster
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Kind: Deploy full stack to local Kubernetes cluster           ║"
	@echo "║  Services: UI + Agent + MCP Server + Postgres + Redis          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@which kind > /dev/null || (echo "Error: kind not found. Install: https://kind.sigs.k8s.io" && exit 1)
	@which kubectl > /dev/null || (echo "Error: kubectl not found." && exit 1)
	@# --- Step 1: Clone MCP server and UI if needed ---
	@if [ ! -d "$(KIND_DIR)/template-mcp-server" ]; then \
		echo "Cloning template-mcp-server (branch: $(KIND_MCP_BRANCH))..."; \
		mkdir -p $(KIND_DIR); \
		git clone --branch $(KIND_MCP_BRANCH) --depth 1 $(KIND_MCP_REPO) $(KIND_DIR)/template-mcp-server; \
	else \
		echo "MCP server already cloned"; \
	fi
	@if [ "$(KIND_UI_DIR)" != "../template-ui" ] && [ ! -d "$(KIND_DIR)/template-ui" ]; then \
		echo "Cloning template-ui (branch: $(KIND_UI_BRANCH))..."; \
		mkdir -p $(KIND_DIR); \
		git clone --branch $(KIND_UI_BRANCH) --depth 1 $(KIND_UI_REPO) $(KIND_DIR)/template-ui; \
	else \
		echo "UI already cloned"; \
	fi
	@# --- Step 2: Create cluster if not exists ---
	@if ! kind get clusters 2>/dev/null | grep -q "$(KIND_CLUSTER)"; then \
		echo "Creating Kind cluster '$(KIND_CLUSTER)'..."; \
		kind create cluster --name $(KIND_CLUSTER) --config=- <<< '{"kind":"Cluster","apiVersion":"kind.x-k8s.io/v1alpha4","nodes":[{"role":"control-plane","kubeadmConfigPatches":["kind: InitConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: ingress-ready=true\n"],"extraPortMappings":[{"containerPort":80,"hostPort":$(KIND_HTTP_PORT),"protocol":"TCP"},{"containerPort":443,"hostPort":$(KIND_HTTPS_PORT),"protocol":"TCP"}]}]}'; \
		echo "Installing NGINX Ingress..."; \
		$(KCTL) apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml; \
		echo "Waiting for ingress controller pod to be scheduled..."; \
		sleep 10; \
		$(KCTL) wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=120s; \
	else \
		echo "Kind cluster '$(KIND_CLUSTER)' already exists"; \
	fi
	@# --- Step 3: Build and load images ---
	@echo "Building agent image..."
	@podman build -t $(KIND_IMAGE) .
	@echo "Building MCP server image..."
	@podman build -t $(KIND_MCP_IMAGE) -f $(KIND_DIR)/template-mcp-server/Containerfile $(KIND_DIR)/template-mcp-server
	@echo "Building UI image from $(KIND_UI_DIR)..."
	@podman build -t $(KIND_UI_IMAGE) $(KIND_UI_DIR)
	@echo "Loading images into Kind (podman -> archive -> kind)..."
	@podman save $(KIND_IMAGE) -o /tmp/kind-agent.tar && kind load image-archive /tmp/kind-agent.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-agent.tar
	@podman save $(KIND_MCP_IMAGE) -o /tmp/kind-mcp.tar && kind load image-archive /tmp/kind-mcp.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-mcp.tar
	@podman save $(KIND_UI_IMAGE) -o /tmp/kind-ui.tar && kind load image-archive /tmp/kind-ui.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-ui.tar
	@$(MAKE) --no-print-directory kind-gc
	@# --- Step 4: Deploy ---
	@echo "Deploying to Kind..."
	@$(KCTL) create namespace $(KIND_NS) 2>/dev/null || true
	@$(KCTL) apply -k deployment/overlays/kind/
	@# Upstream template-mcp-server ships only an openshift overlay now,
	@# so this is conditional rather than fatal: the agent reaches its
	@# real MCP servers over the network, and the example one is a
	@# convenience.
	@if [ -d "$(KIND_DIR)/template-mcp-server/deployment/kind" ]; then \
		$(KCTL) apply -k $(KIND_DIR)/template-mcp-server/deployment/kind/; \
	else \
		echo "  (skipping MCP server: no kind overlay in template-mcp-server)"; \
	fi
	@$(MAKE) --no-print-directory kind-seed-config
	@echo ""
	@echo "Waiting for pods..."

	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=database --timeout=60s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=cache --timeout=60s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=mcp-server --timeout=30s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=agent --timeout=120s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=ui --timeout=90s 2>/dev/null || true
	@# --- Step 6: Port-forwards for localhost access ---
	@echo "Setting up port-forwards..."
	@$(KCTL) -n $(KIND_NS) port-forward svc/ui 8080:8080 &>/dev/null &
	@$(KCTL) -n $(KIND_NS) port-forward svc/agent 5002:5002 &>/dev/null &
	@$(KCTL) -n $(KIND_NS) port-forward svc/mcp-server 5001:5001 &>/dev/null &
	@sleep 2
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Kind cluster ready!                                           ║"
	@echo "║  UI:         http://localhost:8080                             ║"
	@echo "║  Ingress:    http://localhost:$(KIND_HTTP_PORT)                           ║"
	@echo "║  Agent:      http://localhost:5002                             ║"
	@echo "║  MCP Server: http://localhost:5001                             ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Useful commands:"
	@echo "  Pods:      $(KCTL) -n $(KIND_NS) get pods"
	@echo "  Logs:      $(KCTL) -n $(KIND_NS) logs -l component=agent -f"
	@echo "  Teardown:  make kind-down"

kind-down: ## Delete the Kind cluster and clean up cloned repos
	@echo "Stopping port-forwards..."
	@pkill -f "kubectl.*port-forward.*$(KIND_NS)" 2>/dev/null || true
	@echo "Deleting Kind cluster '$(KIND_CLUSTER)'..."
	@kind delete cluster --name $(KIND_CLUSTER) 2>/dev/null || true
	@rm -rf $(KIND_DIR)
	@echo "Kind cluster and .kind/ cleaned up."

undeploy:
	@if [ "$(filter openshift,$(MAKECMDGOALS))" = "openshift" ]; then \
		echo "Checking for oc CLI..."; \
		which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1); \
		oc project $(NAMESPACE) || (echo "Error: Cannot switch to namespace '$(NAMESPACE)'" && exit 1); \
		echo "Removing OpenShift deployment..."; \
		oc delete deployment,service,route,configmap,secret,pvc,buildconfig,imagestream -l app=agent 2>/dev/null || true; \
		echo "Undeployment complete!"; \
	elif [ "$(filter mpp,$(MAKECMDGOALS))" = "mpp" ]; then \
		echo "Checking for oc CLI..."; \
		RUNTIME_NAMESPACE="$(TENANT)--template"; \
		which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1); \
		oc project $$RUNTIME_NAMESPACE || (echo "Error: Cannot switch to runtime namespace '$$RUNTIME_NAMESPACE'" && exit 1); \
		echo "Removing MPP deployment..."; \
		oc delete deployment,service,route,configmap,secret,pvc,buildconfig,imagestream -l app=agent 2>/dev/null || true; \
		echo "Undeployment complete!"; \
	else \
		echo "Usage: make undeploy [openshift|mpp]"; \
		echo "Available undeployment targets: openshift, mpp"; \
		exit 1; \
	fi

# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------

test-triggers: ## Unit tests for triggers only
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	.venv/bin/python -m pytest tests/unit/triggers -v

test-integration: ## Integration tests (requires Redis + DB)
	@if [ ! -d ".venv" ]; then \
		echo "Error: Virtual environment not found. Run 'make install' first to set up the environment."; \
		exit 1; \
	fi
	@echo "Running integration tests..."
	.venv/bin/python -m pytest tests/integration -m integration -v

%:
	@:

validate-config: ## Validate config/agent against its models; non-zero on error
	.venv/bin/python -m deep_agent.src.agent.config.validate

whats-there: ## Before scoping work: does it already exist? make whats-there Q=queue
	@test -n "$(Q)" || (echo "usage: make whats-there Q=<keyword>" && exit 2)
	@.venv/bin/python scripts/whats_there.py "$(Q)" 2>/dev/null

record-cassettes: ## Re-record LLM cassettes against the real provider (costs money)
	@echo "Recording against live providers — requires credentials in .env"
	CASSETTE_MODE=record .venv/bin/python -m pytest $(ARGS) -q

openapi-baseline: ## Regenerate the committed OpenAPI contract baseline
	@.venv/bin/python -c "import json,io,contextlib; \
		buf=io.StringIO(); \
		ctx=contextlib.redirect_stderr(buf); ctx.__enter__(); \
		from aegra_api.main import create_app; s=create_app().openapi(); ctx.__exit__(None,None,None); \
		io.open('tests/contracts/openapi-baseline.json','w').write(json.dumps(s,indent=2,sort_keys=True))" 2>/dev/null
	@echo "Wrote tests/contracts/openapi-baseline.json — review the diff before committing"

test-contract: ## Cross-repo and cross-version interface tests (per PR)
	.venv/bin/python -m pytest tests/unit/aegra/test_ui_contract.py tests/unit/aegra/test_openapi_contract.py -q

test-nightly: ## Kill-and-restart tier; needs a running stack
	.venv/bin/python -m pytest tests/nightly -m nightly -v

ADC_FILE ?= $(HOME)/.config/gcloud/application_default_credentials.json
# gcloud's configured project if there is one, else the ADC's quota
# project — a user credential usually carries the latter and nothing else.
GCP_PROJECT ?= $(shell gcloud config get-value project 2>/dev/null | grep -v '^(unset)$$' || true)
GCP_PROJECT_RESOLVED = $(or $(strip $(GCP_PROJECT)),$(shell python3 -c "import json,os;print(json.load(open(os.path.expanduser('$(ADC_FILE)'))).get('quota_project_id',''))" 2>/dev/null))

kind-vertex: ## Point the Kind agent at real Vertex AI instead of the mock model
	@test -f "$(ADC_FILE)" || (echo "No ADC file at $(ADC_FILE). Run: gcloud auth application-default login" && exit 1)
	@# --from-file, so the credential goes straight from disk into the
	@# Secret. It is never interpolated into a command line, where it
	@# would land in shell history and process listings.
	@$(KCTL) -n $(KIND_NS) create secret generic google-adc --from-file=application_default_credentials.json="$(ADC_FILE)" --dry-run=client -o yaml | $(KCTL) -n $(KIND_NS) apply -f - >/dev/null
	@# A strategic merge, not a JSON patch. `add` on `/volumes` replaces the
	@# whole array, which dropped the `agent-documents` PVC mount for as long
	@# as Vertex was active. Merging by the `name` key appends when absent and
	@# updates when present, so this is also safe to run twice.
	@$(KCTL) -n $(KIND_NS) patch deploy/agent -p '{"spec":{"template":{"spec":{"volumes":[{"name":"google-adc","secret":{"secretName":"google-adc"}}],"containers":[{"name":"agent","volumeMounts":[{"name":"google-adc","mountPath":"/var/run/gcp","readOnly":true}]}]}}}}' >/dev/null
	@# Empty strings, not `VAR-`: these come from the ConfigMap through
	@# envFrom, and `set env VAR-` only drops an explicit container entry.
	@# An explicit empty value is what actually shadows an envFrom one —
	@# which is why the first attempt at this silently kept using the stub.
	@$(KCTL) -n $(KIND_NS) set env deploy/agent GOOGLE_APPLICATION_CREDENTIALS=/var/run/gcp/application_default_credentials.json GOOGLE_CLOUD_PROJECT=$(GCP_PROJECT_RESOLVED) MODEL_OVERRIDE="" VLLM_BASE_URL="" >/dev/null
	@$(KCTL) -n $(KIND_NS) rollout status deploy/agent --timeout=300s
	@echo "Agent now using Vertex AI (project: $(GCP_PROJECT_RESOLVED)). Revert with 'make kind-mock'."

kind-mock: ## Point the Kind agent back at the in-cluster mock model
	@# Delete the one entry by merge key. Removing the whole array took the
	@# `agent-documents` mount with it until `apply -k` put it back.
	@$(KCTL) -n $(KIND_NS) patch deploy/agent -p '{"spec":{"template":{"spec":{"volumes":[{"name":"google-adc","$$patch":"delete"}],"containers":[{"name":"agent","volumeMounts":[{"name":"google-adc","$$patch":"delete"}]}]}}}}' >/dev/null 2>&1 || true
	@$(KCTL) -n $(KIND_NS) set env deploy/agent \
		GOOGLE_APPLICATION_CREDENTIALS- GOOGLE_CLOUD_PROJECT- \
		MODEL_OVERRIDE- VLLM_BASE_URL- >/dev/null
	@$(KCTL) -n $(KIND_NS) apply -k deployment/overlays/kind/ >/dev/null
	@$(KCTL) -n $(KIND_NS) rollout status deploy/agent --timeout=300s
	@$(KCTL) -n $(KIND_NS) delete secret google-adc --ignore-not-found >/dev/null
	@echo "Agent back on the mock model."

test-nightly-kind: ## Kill-and-restart tier against the local Kind cluster (make kind first)
	@kind get clusters 2>/dev/null | grep -q "$(KIND_CLUSTER)" || \
		(echo "No Kind cluster '$(KIND_CLUSTER)'. Run 'make kind' first." && exit 1)
	@# Through the ingress, not a port-forward: a forward binds to one pod and
	@# dies with it, and every assertion after the kill then fails in a way
	@# indistinguishable from a recovery failure.
	@# The tier asks the stub for a slow response per request (see
	@# tests/nightly), so running it does not slow the cluster down for
	@# anyone else using the UI at the time.
	@# `\$$(` in WORKER_KILL_COMMAND so the pod lookup runs each time the
	@# command is executed. Left unescaped the shell resolved it once, and the
	@# second kill in a run targeted a pod that was already gone.
	@AGENT_URL=http://agent.localhost:$(KIND_HTTP_PORT) \
	WORKER_KILL_COMMAND="$(KCTL) -n $(KIND_NS) delete \$$($(KCTL) -n $(KIND_NS) get pod -l component=agent -o name | head -1) --force --grace-period=0" \
	.venv/bin/python -m pytest tests/nightly -m nightly -v

test-deployment: ## Render every overlay and assert the OpenShift constraints
	.venv/bin/python -m pytest tests/unit/deployment -q

render-deployment: ## Regenerate deployment overlays from config/agent/deployment/values.yaml
	.venv/bin/python scripts/render_deployment.py

render-deployment-check: ## Fail if the overlays have drifted from values.yaml (CI)
	.venv/bin/python scripts/render_deployment.py --check

dataset-skill: ## Turn a dataset into a skill: make dataset-skill NAME="sales" SRC=./data
	@test -n "$(NAME)" -a -n "$(SRC)" || (echo 'Usage: make dataset-skill NAME="sales" SRC=./data' && exit 1)
	.venv/bin/python scripts/dataset_skill.py --name "$(NAME)" --src "$(SRC)" $(if $(FORCE),--force)

config-reference: ## Regenerate docs/config-reference.md from the Pydantic models
	@.venv/bin/python scripts/config_reference.py 2>/dev/null

new-agent: ## Scaffold a clean config/ for a new agent: make new-agent NAME="My Agent"
	@test -n "$(NAME)" || (echo 'usage: make new-agent NAME="My Agent" [OUT=path]' && exit 2)
	@.venv/bin/python scripts/new_agent.py --name "$(NAME)" $(if $(OUT),--out "$(OUT)") 2>/dev/null

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kind image housekeeping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# `kind load image-archive` imports the whole image under a dated
# `docker.io/library/import-<date>` ref and takes out a containerd
# **lease** on its content. Neither is ever cleaned up, so every load
# of a ~3 GB image is permanently retained.
#
# This filled the Kind node to 176 GB across 150 orphaned imports and
# broke three separate deploys with "no space left on device" — once
# silently, where `podman save` failed, `kubectl rollout status` still
# reported success, and the cluster quietly kept running the previous
# image.
#
# Deleting the image refs alone reclaims nothing: the leases still
# pin the content. Both have to go, in that order.
#
# The ref is `import-<date>@sha256:...` — note there is no
# `docker.io/library/` prefix, even though `crictl images` prints one.
# Matching crictl's display form instead of ctr's real ref made this
# target silently reclaim nothing for several days while reporting
# success, and the node grew to 147 GB.
.PHONY: kind-gc
kind-gc:
	@echo "Reclaiming Kind node image space..."
	@podman exec $(KIND_CLUSTER)-control-plane sh -c '\
	  for i in $$(ctr -n k8s.io images ls -q 2>/dev/null | grep "^import-"); do \
	    ctr -n k8s.io images rm "$$i" >/dev/null 2>&1 || true; \
	  done; \
	  for l in $$(ctr -n k8s.io leases ls -q 2>/dev/null); do \
	    ctr -n k8s.io leases rm "$$l" >/dev/null 2>&1 || true; \
	  done' 2>/dev/null || true
	@podman exec $(KIND_CLUSTER)-control-plane df -h /var 2>/dev/null | tail -1 || true

.PHONY: kind-seed-config
kind-seed-config: ## Copy agent and UI config onto the config volume
	@echo "Seeding config volumes..."
	@# A pod rather than `kubectl cp` straight to the agent: the agent
	@# is still in its init container waiting for exactly this, so
	@# there is nothing to copy into yet.
	@$(KCTL) -n $(KIND_NS) delete pod config-seeder --ignore-not-found --wait=true >/dev/null 2>&1 || true
	@$(KCTL) -n $(KIND_NS) apply -f deployment/overlays/kind/config-seeder.yaml >/dev/null
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod/config-seeder --timeout=120s >/dev/null
	@# One claim with two subtrees. Clear the *contents*, never the
	@# directories: a subPath mount binds the directory's inode, so
	@# `rm -rf /seed/agent` leaves every pod that already mounted it
	@# bound to a deleted inode, reading an empty tree forever.
	@$(KCTL) -n $(KIND_NS) exec config-seeder -- sh -c 'mkdir -p /seed/agent /seed/ui && rm -rf /seed/agent/* /seed/ui/*' >/dev/null
	@$(KCTL) -n $(KIND_NS) cp config/agent/. config-seeder:/seed/agent/ >/dev/null
	@# From the same checkout the UI image is built from — the agent
	@# repo has no config/ui, and seeding the two from different
	@# places is how they drift.
	@$(KCTL) -n $(KIND_NS) cp $(KIND_UI_DIR)/config/ui/settings.yaml config-seeder:/seed/ui/ui.yaml >/dev/null
	@$(KCTL) -n $(KIND_NS) exec config-seeder -- sh -c 'chmod -R a+rX /seed' >/dev/null
	@echo "  agent: $$($(KCTL) -n $(KIND_NS) exec config-seeder -- sh -c 'ls /seed/agent | tr "\n" " "')"
	@echo "  ui:    $$($(KCTL) -n $(KIND_NS) exec config-seeder -- sh -c 'ls /seed/ui | tr "\n" " "')"
	@$(KCTL) -n $(KIND_NS) delete pod config-seeder --wait=false >/dev/null
	@echo "Config volumes seeded."

.PHONY: kind-reseed
kind-reseed: kind-seed-config ## Re-seed config and restart the pods that read it
	@$(KCTL) -n $(KIND_NS) rollout restart deployment/agent deployment/ui >/dev/null
	@$(KCTL) -n $(KIND_NS) rollout status deployment/agent --timeout=180s
	@$(KCTL) -n $(KIND_NS) rollout status deployment/ui --timeout=120s

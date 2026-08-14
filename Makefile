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
	echo "Building container image from source..."; \
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
	echo "Deploying resources to OpenShift..."; \
	oc apply -k deployment/overlays/openshift/ || (mv deployment/overlays/openshift/kustomization.yaml.bak deployment/overlays/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	rm -f deployment/overlays/openshift/kustomization.yaml.bak; \
	echo "Deployment complete!"; \
	echo "Checking deployment status..."; \
	oc get pods -l app=agent; \
	echo ""; \
	echo "Useful commands:"; \
	echo "  View logs: oc logs -l app=agent --tail=100"; \
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
KIND_MCP_IMAGE := template-mcp-server:local
KIND_UI_IMAGE := template-ui:local
KIND_NS := template-agent
KIND_DIR := .kind
KIND_MCP_REPO := https://github.com/redhat-data-and-ai/template-mcp-server.git
KIND_MCP_BRANCH := feat/rh-flavour
KIND_UI_REPO := https://github.com/redhat-data-and-ai/template-ui.git
KIND_UI_BRANCH := feat/rh-flavour
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
	@if [ ! -d "$(KIND_DIR)/template-ui" ]; then \
		echo "Cloning template-ui (branch: $(KIND_UI_BRANCH))..."; \
		mkdir -p $(KIND_DIR); \
		git clone --branch $(KIND_UI_BRANCH) --depth 1 $(KIND_UI_REPO) $(KIND_DIR)/template-ui; \
	else \
		echo "UI already cloned"; \
	fi
	@# --- Step 2: Create cluster if not exists ---
	@if ! kind get clusters 2>/dev/null | grep -q "$(KIND_CLUSTER)"; then \
		echo "Creating Kind cluster '$(KIND_CLUSTER)'..."; \
		kind create cluster --name $(KIND_CLUSTER) --config=- <<< '{"kind":"Cluster","apiVersion":"kind.x-k8s.io/v1alpha4","nodes":[{"role":"control-plane","kubeadmConfigPatches":["kind: InitConfiguration\nnodeRegistration:\n  kubeletExtraArgs:\n    node-labels: ingress-ready=true\n"],"extraPortMappings":[{"containerPort":80,"hostPort":80,"protocol":"TCP"},{"containerPort":443,"hostPort":443,"protocol":"TCP"}]}]}'; \
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
	@echo "Building UI image..."
	@podman build -t $(KIND_UI_IMAGE) $(KIND_DIR)/template-ui
	@echo "Loading images into Kind (podman -> archive -> kind)..."
	@podman save $(KIND_IMAGE) -o /tmp/kind-agent.tar && kind load image-archive /tmp/kind-agent.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-agent.tar
	@podman save $(KIND_MCP_IMAGE) -o /tmp/kind-mcp.tar && kind load image-archive /tmp/kind-mcp.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-mcp.tar
	@podman save $(KIND_UI_IMAGE) -o /tmp/kind-ui.tar && kind load image-archive /tmp/kind-ui.tar --name $(KIND_CLUSTER) && rm -f /tmp/kind-ui.tar
	@# --- Step 4: Deploy ---
	@echo "Deploying to Kind..."
	@$(KCTL) create namespace $(KIND_NS) 2>/dev/null || true
	@$(KCTL) apply -k deployment/overlays/kind/
	@$(KCTL) apply -k $(KIND_DIR)/template-mcp-server/deployment/kind/
	@echo ""
	@echo "Waiting for pods..."
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=database --timeout=60s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=cache --timeout=60s 2>/dev/null || true
	@$(KCTL) -n $(KIND_NS) wait --for=condition=ready pod -l component=mcp-server --timeout=90s 2>/dev/null || true
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

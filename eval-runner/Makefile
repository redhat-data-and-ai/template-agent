## Eval runner Makefile
##
## Builds and runs the eval runner service (separate pod, not a sidecar).
## Requires the root stack (pgvector + template-agent) to be running first.
##
##   make local        - Run via venv (fast, no container build)
##   make up           - Build and start as container (joins root stack network)
##   make down         - Stop container
##   make logs         - Tail container logs
##   make build        - Build image only
##   make venv         - Create isolated venv

.PHONY: local up down logs build venv test deploy

COMPOSE = podman-compose -f compose.yaml

# ── Venv ────────────────────────────────────────────────────────────────────

.venv/.installed:
	@test -d .venv || uv venv .venv --python 3.12
	@uv pip install -q --python .venv/bin/python \
		"lightspeed-evaluation @ git+https://github.com/lightspeed-core/lightspeed-evaluation.git@v0.8.0" \
		fastapi "uvicorn[standard]" httpx pyyaml psycopg2-binary
	@touch .venv/.installed

venv: .venv/.installed

test: venv ## Run unit tests with coverage
	@uv pip install -q --python .venv/bin/python \
		pytest pytest-asyncio pytest-cov httpx
	@EVAL_INTERNAL_TOKEN=test-token PYTHONPATH=src .venv/bin/pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=90 -v

# ── Local (venv) ─────────────────────────────────────────────────────────────

local: venv ## Run via venv — fast local dev, requires make local in parent dir
	@echo "Starting eval runner on http://localhost:8099"
	@echo "Requires: agent running at http://127.0.0.1:5002"
	@echo "Press Ctrl+C to stop"
	@trap 'lsof -ti :8099 | xargs kill -INT 2>/dev/null || true; sleep 1; lsof -ti :8099 | xargs kill -9 2>/dev/null || true; exit 130' INT TERM; \
	set -a; [ -f ../.env ] && . ../.env 2>/dev/null; set +a; \
	DEPLOYED_AGENT_ORG=demo \
	DEPLOYED_AGENT_NAME=template-agent \
	POSTGRES_HOST=localhost \
	POSTGRES_PORT=5432 \
	POSTGRES_DB=template_agent \
	POSTGRES_USER=postgres \
	POSTGRES_PASSWORD=postgres \
	AGENT_HOST=http://127.0.0.1:5002 \
	AGENT_CONFIG_DIR=../config/agent \
	PYTHONPATH=src \
	.venv/bin/uvicorn eval_api:app --port 8099

# ── Container ────────────────────────────────────────────────────────────────

build: ## Build container image
	podman build -f Containerfile -t template-agent-eval-runner:local .

up: ## Build and start eval runner container (joins root stack network)
	$(COMPOSE) --profile eval up --build -d

down: ## Stop eval runner container
	$(COMPOSE) --profile eval down

logs: ## Tail eval runner container logs
	$(COMPOSE) --profile eval logs -f eval-runner

# ── OpenShift ────────────────────────────────────────────────────────────────

deploy: ## Deploy eval runner to OpenShift (requires: NAMESPACE)
	@echo "Checking for oc CLI..."
	@which oc > /dev/null || (echo "Error: oc CLI not found." && exit 1)
	@if [ -z "$(NAMESPACE)" ]; then \
		echo "Error: NAMESPACE not set. Usage: make deploy NAMESPACE=your-project"; \
		exit 1; \
	fi; \
	echo "Using namespace: $(NAMESPACE)"; \
	oc project $(NAMESPACE) || (echo "Error: Cannot switch to namespace '$(NAMESPACE)'" && exit 1); \
	echo "Applying eval-runner manifests..."; \
	oc apply -f deployment/overlays/openshift/eval-runner-imagestream.yaml; \
	oc apply -f deployment/overlays/openshift/eval-runner-buildconfig.yaml; \
	oc apply -f deployment/components/eval-runner/service.yaml; \
	oc apply -f deployment/components/eval-runner/networkpolicy.yaml; \
	echo "Building eval-runner image..."; \
	oc start-build eval-runner --from-dir=. \
		--exclude='(^|/)\.venv(/|$$)' \
		--exclude='(^|/)__pycache__(/|$$)' \
		--follow || exit 1; \
	echo ""; \
	echo "Useful commands:"; \
	echo "  Logs:   oc logs -l app=eval-runner --tail=100"; \
	echo "  Status: oc get pods -l app=eval-runner"

.PHONY: local dev test clean deploy undeploy

# OpenShift namespace (can be overridden: make deploy openshift NAMESPACE=my-project)
NAMESPACE ?= $(shell oc project -q 2>/dev/null)

local:
	@echo "Setting up and starting Template Agent locally..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi
	@echo "Activating virtual environment..."
	@source .venv/bin/activate && \
	echo "Installing dependencies..." && \
	uv pip install -e ".[dev]" && \
	echo "Starting with in-memory storage..." && \
	USE_INMEMORY_SAVER=true python -m template_agent.src.main

dev:
	@echo "Starting Template Agent with podman-compose..."
	@podman-compose up --build

test:
	@echo "Running tests..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi
	@source .venv/bin/activate && \
	echo "Installing dependencies..." && \
	uv pip install -e ".[dev]" && \
	echo "Running pytest..." && \
	pytest

clean:
	@echo "Cleaning up non-code artifacts..."
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
	sed -i.bak "s|NAMESPACE_PLACEHOLDER|$(NAMESPACE)|g" deployment/openshift/deployment.yaml; \
	sed -i.bak "s|namespace: template-agent|namespace: $(NAMESPACE)|g" deployment/openshift/kustomization.yaml; \
	echo "Creating BuildConfig and ImageStream..."; \
	oc apply -f deployment/openshift/buildconfig.yaml; \
	oc apply -f deployment/openshift/imagestream.yaml; \
	echo "Building container image from source..."; \
	oc start-build template-agent --from-dir=. --follow || (mv deployment/openshift/deployment.yaml.bak deployment/openshift/deployment.yaml 2>/dev/null; mv deployment/openshift/kustomization.yaml.bak deployment/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	echo "Deploying resources to OpenShift..."; \
	oc apply -k deployment/openshift/ || (mv deployment/openshift/deployment.yaml.bak deployment/openshift/deployment.yaml 2>/dev/null; mv deployment/openshift/kustomization.yaml.bak deployment/openshift/kustomization.yaml 2>/dev/null; exit 1); \
	rm -f deployment/openshift/deployment.yaml.bak deployment/openshift/kustomization.yaml.bak; \
	echo "Deployment complete!"; \
	echo "Checking deployment status..."; \
	oc get pods -l app=template-agent; \
	echo ""; \
	echo "Useful commands:"; \
	echo "  View logs: oc logs -l app=template-agent --tail=100"; \
	echo "  Get route: oc get route template-agent"; \
	echo "  Check status: oc get pods,svc,route -l app=template-agent"

mpp:
	@echo "Checking for oc CLI..."
	@which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1)
	@echo "Validating TENANT parameter..."
	@if [ -z "$(TENANT)" ]; then \
		echo "Error: TENANT not set. Usage: make deploy mpp TENANT=your-tenant"; \
		exit 1; \
	fi; \
	CONFIG_NAMESPACE="$(TENANT)--config"; \
	RUNTIME_NAMESPACE="$(TENANT)--template"; \
	echo "Config namespace: $$CONFIG_NAMESPACE"; \
	echo "Runtime namespace: $$RUNTIME_NAMESPACE"; \
	echo "Updating tenant.yaml with config namespace..."; \
	sed -i.bak "s|TENANT_PLACEHOLDER|$$CONFIG_NAMESPACE|g" deployment/mpp/tenant.yaml; \
	echo "Creating/switching to config namespace..."; \
	oc project $$CONFIG_NAMESPACE 2>/dev/null || oc new-project $$CONFIG_NAMESPACE || (echo "Error: Cannot create/switch to namespace '$$CONFIG_NAMESPACE'." && mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null && exit 1); \
	echo "Applying TenantNamespace CR to create runtime namespace..."; \
	oc apply -f deployment/mpp/tenant.yaml || (mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null && exit 1); \
	echo "Waiting for runtime namespace '$$RUNTIME_NAMESPACE' to be created..."; \
	COUNTER=1; \
	until oc get project $$RUNTIME_NAMESPACE 2>/dev/null || [ $$COUNTER -gt 30 ]; do \
		echo "Waiting for namespace... ($$COUNTER/30)"; \
		sleep 2; \
		COUNTER=$$((COUNTER + 1)); \
	done; \
	if [ $$COUNTER -le 30 ]; then \
		echo "Runtime namespace '$$RUNTIME_NAMESPACE' is ready"; \
	fi; \
	oc project "$(TENANT)--$(RUNTIME_NAMESPACE)" > /dev/null 2>&1 || (echo "Error: Runtime namespace '$$RUNTIME_NAMESPACE' was not created" && mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null && exit 1); \
	echo "Switching to runtime namespace..."; \
	oc project $$RUNTIME_NAMESPACE || (echo "Error: Cannot switch to runtime namespace '$$RUNTIME_NAMESPACE'" && mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null && exit 1); \
	echo "Creating BuildConfig and ImageStream..."; \
	oc apply -f deployment/mpp/buildconfig.yaml; \
	oc apply -f deployment/mpp/imagestream.yaml; \
	echo "Building container image from source..."; \
	oc start-build template-agent --from-dir=. --follow || (mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null; exit 1); \
	echo "Deploying resources to MPP..."; \
	oc apply -k deployment/mpp/ || (mv deployment/mpp/tenant.yaml.bak deployment/mpp/tenant.yaml 2>/dev/null; exit 1); \
	rm -f deployment/mpp/tenant.yaml.bak; \
	echo "Deployment complete!"; \
	echo "Checking deployment status..."; \
	oc get pods -l app=template-agent; \
	echo ""; \
	echo "Useful commands:"; \
	echo "  View logs: oc logs -l app=template-agent --tail=100"; \
	echo "  Get route: oc get route template-agent"; \
	echo "  Check status: oc get pods,svc,route -l app=template-agent"

undeploy:
	@if [ "$(filter openshift,$(MAKECMDGOALS))" = "openshift" ]; then \
		echo "Checking for oc CLI..."; \
		which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1); \
		echo "Removing OpenShift deployment..."; \
		oc delete deployment,service,route,configmap,secret,pvc,buildconfig,imagestream -l app=template-agent 2>/dev/null || true; \
		echo "Undeployment complete!"; \
	elif [ "$(filter mpp,$(MAKECMDGOALS))" = "mpp" ]; then \
		echo "Checking for oc CLI..."; \
		which oc > /dev/null || (echo "Error: oc CLI not found. Please install OpenShift CLI." && exit 1); \
		echo "Removing MPP deployment..."; \
		oc delete deployment,service,route,configmap,secret,pvc,buildconfig,imagestream -l app=template-agent 2>/dev/null || true; \
		echo "Undeployment complete!"; \
	else \
		echo "Usage: make undeploy [openshift|mpp]"; \
		echo "Available undeployment targets: openshift, mpp"; \
		exit 1; \
	fi

%:
	@:

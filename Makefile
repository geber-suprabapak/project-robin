# ============================================================================
# Project Robin - Build Automation
# ============================================================================

# Registry Config
# Try to grep OWNER from .env (GHCR_USERNAME), fallback to default
ENV_OWNER := $(shell grep "^GHCR_USERNAME=" .env 2>/dev/null | cut -d= -f2- | sed 's/[ #].*//' | tr -d '"' | tr -d "'")

REGISTRY ?= ghcr.io
OWNER ?= $(if $(ENV_OWNER),$(ENV_OWNER),geber-suprabapak)
REPO ?= project-robin
IMAGE_NAME ?= $(REGISTRY)/$(OWNER)/$(REPO)

# Versions
TAG ?= latest

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Configuration:"
	@echo "  Registry: $(REGISTRY)"
	@echo "  Owner:    $(OWNER)"
	@echo "  Repo:     $(REPO)"
	@echo ""
	@echo "Targets:"
	@echo "  build-cpu       Build CPU Docker image (Default)"
	@echo "  build-gpu       Build GPU Docker image"
	@echo "  push-cpu        Build and push CPU image to GHCR (Auto-logins using .env)"
	@echo "  push-gpu        Build and push GPU image to GHCR (Auto-logins using .env)"
	@echo "  up-cpu          Start CPU services via Docker Compose"
	@echo "  up-gpu          Start GPU services via Docker Compose"
	@echo "  down            Stop all services"
	@echo "  clean           Remove local images and containers"

# ----------------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------------

.PHONY: login
login: ## Login to Docker Registry using .env credentials
	@echo "🔐 Checking registry credentials..."
	@if [ -f .env ]; then \
		TOKEN=$$(grep "^GHCR_TOKEN=" .env | cut -d= -f2- | sed 's/[ #].*//' | tr -d '"' | tr -d "'"); \
		USER=$$(grep "^GHCR_USERNAME=" .env | cut -d= -f2- | sed 's/[ #].*//' | tr -d '"' | tr -d "'"); \
		if [ -n "$$TOKEN" ] && [ -n "$$USER" ]; then \
			echo "   Logging in to $(REGISTRY) as $$USER..."; \
			echo "$$TOKEN" | docker login $(REGISTRY) -u "$$USER" --password-stdin; \
		else \
			echo "   ⚠ GHCR credentials not found in .env (GHCR_TOKEN, GHCR_USERNAME)"; \
		fi \
	else \
		echo "   ⚠ .env file not found"; \
	fi

# ----------------------------------------------------------------------------
# Build Commands
# ----------------------------------------------------------------------------

.PHONY: build
build: build-cpu ## Alias for build-cpu

.PHONY: build-cpu
build-cpu: ## Build CPU image
	@echo "🐳 Building CPU Image [$(IMAGE_NAME):cpu-$(TAG)]..."
	docker build \
		-t $(IMAGE_NAME):cpu-$(TAG) \
		-t project-robin:cpu-latest \
		--build-arg RUNTIME_TYPE=cpu \
		--build-arg BASE_IMAGE=python:3.12-slim-bookworm \
		-f docker/Dockerfile .

.PHONY: build-gpu
build-gpu: ## Build GPU image
	@echo "🐳 Building GPU Image [$(IMAGE_NAME):$(TAG)]..."
	docker build \
		-t $(IMAGE_NAME):$(TAG) \
		-t project-robin:latest \
		--build-arg RUNTIME_TYPE=gpu \
		--build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
		-f docker/Dockerfile .

# ----------------------------------------------------------------------------
# Push Commands (Depends on login)
# ----------------------------------------------------------------------------

.PHONY: push-cpu
push-cpu: login build-cpu ## Push CPU image
	@echo "🚀 Pushing CPU Image to $(IMAGE_NAME):cpu-$(TAG)..."
	docker push $(IMAGE_NAME):cpu-$(TAG)

.PHONY: push-gpu
push-gpu: login build-gpu ## Push GPU image
	@echo "🚀 Pushing GPU Image to $(IMAGE_NAME):$(TAG)..."
	docker push $(IMAGE_NAME):$(TAG)

# ----------------------------------------------------------------------------
# Run Commands
# ----------------------------------------------------------------------------

.PHONY: up-cpu
up-cpu: ## Start CPU Stack
	docker compose --profile cpu up -d

.PHONY: up-gpu
up-gpu: ## Start GPU Stack
	docker compose --profile gpu up -d

.PHONY: down
down: ## Stop Stack
	docker compose down

.PHONY: clean
clean: ## Clean up
	docker rmi project-robin:cpu-latest project-robin:latest || true

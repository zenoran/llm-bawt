# ─────────────────────────────────────────────────────────────────
# llm-bawt Makefile — cross-platform (Linux / macOS / Windows)
# ─────────────────────────────────────────────────────────────────
#
# Usage:
#   make help            Show available targets
#   make install         Install via pipx (production)
#   make dev             Sync local .venv with all extras (development)
#   make up              Docker compose up (production)
#   make docker-dev      Docker compose up (dev mode, live mount)
#   make server-start    Start MCP + LLM services locally
#   make test            Run tests
#   make lint            Lint + format check
#
# ─────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help

# ── OS detection ────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    DETECTED_OS  := Windows
    SHELL        := cmd.exe
    PYTHON       := python
    RM_RF        := rmdir /s /q
    MKDIR        := mkdir
    NULL         := NUL
    SEP          := \\
    # Windows: no color by default
    BLUE         :=
    GREEN        :=
    YELLOW       :=
    RED          :=
    NC           :=
    AND          := &
    CHECK_CMD     = where $(1) >$(NULL) 2>&1
    FIND_PID      = $(error PID-based service management not supported on Windows — use Docker)
else
    DETECTED_OS  := $(shell uname -s)
    PYTHON       := python3
    RM_RF        := rm -rf
    MKDIR         = mkdir -p
    NULL         := /dev/null
    SEP          := /
    BLUE         := \033[0;34m
    GREEN        := \033[0;32m
    YELLOW       := \033[1;33m
    RED          := \033[0;31m
    NC           := \033[0m
    AND          := &&
    CHECK_CMD     = command -v $(1) >$(NULL) 2>&1
    FIND_PID      = cat $(1) 2>$(NULL)
endif

# ── Configurable variables ──────────────────────────────────────
REPO           ?= git+https://github.com/zenoran/llm-bawt.git
CUDA_ARCHS     ?= 75;80;86;89;90;120
WITH_CUDA      ?= true

# Docker
COMPOSE_BASE   := docker-compose.yml
COMPOSE_DEV    := docker-compose.dev.yml

# Local servers
MEMORY_HOST    ?= 0.0.0.0
MEMORY_PORT    ?= 8001
SERVICE_HOST   ?= 0.0.0.0
SERVICE_PORT   ?= 8642
MEMORY_URL     := http://127.0.0.1:$(MEMORY_PORT)
RUN_DIR        := .run
LOG_DIR        := .logs
MEMORY_PID     := $(RUN_DIR)/memory-server.pid
SERVICE_PID    := $(RUN_DIR)/llm-service.pid

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "llm-bawt — available targets"
	@echo "════════════════════════════════════════════════════"
ifeq ($(DETECTED_OS),Windows)
	@findstr /R /C:"^[a-zA-Z_-]*:.*##" $(MAKEFILE_LIST)
else
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
endif

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTALLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: check-python check-uv check-pipx

check-python: ## Verify Python 3.12+
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" \
		|| (echo "$(RED)Python 3.12+ required$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python OK$(NC)"

check-uv: ## Verify uv is installed
	@$(call CHECK_CMD,uv) || (echo "$(RED)uv not found — install: https://docs.astral.sh/uv/$(NC)" && exit 1)
	@echo "$(GREEN)✓ uv OK$(NC)"

check-pipx: ## Verify pipx is installed
	@$(call CHECK_CMD,pipx) || (echo "$(RED)pipx not found — install: $(PYTHON) -m pip install --user pipx$(NC)" && exit 1)
	@echo "$(GREEN)✓ pipx OK$(NC)"

.PHONY: install install-local install-local-all uninstall

install: check-python check-pipx ## [pipx] Production install from GitHub
	@echo "$(BLUE)Installing llm-bawt via pipx...$(NC)"
	-pipx uninstall llm-bawt 2>$(NULL) || true
	pipx install --force "$(REPO)"
	pipx runpip llm-bawt install sentence-transformers
	@echo "$(GREEN)✓ llm-bawt installed$(NC)"
	@echo "  Run: llm --status"

install-local: check-python check-pipx ## [pipx] Editable install from local dir (minimal)
	@echo "$(BLUE)Installing llm-bawt (editable) from local dir...$(NC)"
	-pipx uninstall llm-bawt 2>$(NULL) || true
	pipx install --editable ".[memory]"
	@echo "$(GREEN)✓ llm-bawt installed (editable)$(NC)"

install-local-all: check-python check-pipx ## [pipx] Editable install + ALL optional deps
	@echo "$(BLUE)Installing llm-bawt (editable) with all extras...$(NC)"
	-pipx uninstall llm-bawt 2>$(NULL) || true
	pipx install --editable ".[memory,service,search]"
	pipx runpip llm-bawt install transformers torch huggingface-hub accelerate
	pipx runpip llm-bawt install ddgs tavily-python
	pipx runpip llm-bawt install fastapi "uvicorn[standard]" httpx
	@echo "$(GREEN)✓ llm-bawt installed (editable, all extras)$(NC)"
	@echo "  Run: make install-extras-llama  to add llama-cpp (CUDA-aware)"

install-extras-llama: check-pipx ## [pipx] Add llama-cpp-python (with CUDA if available)
ifeq ($(WITH_CUDA),true)
	@echo "$(BLUE)Installing llama-cpp-python (CUDA)...$(NC)"
	CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" \
		pipx runpip llm-bawt install llama-cpp-python --force-reinstall --no-cache-dir
else
	@echo "$(BLUE)Installing llama-cpp-python (CPU)...$(NC)"
	pipx runpip llm-bawt install llama-cpp-python
endif
	@echo "$(GREEN)✓ llama-cpp-python installed$(NC)"

install-extras-hf: check-pipx ## [pipx] Add HuggingFace deps
	@echo "$(BLUE)Installing HuggingFace deps...$(NC)"
	pipx runpip llm-bawt install transformers torch huggingface-hub accelerate
	@echo "$(GREEN)✓ HuggingFace deps installed$(NC)"

install-extras-service: check-pipx ## [pipx] Add FastAPI service deps
	@echo "$(BLUE)Installing service deps...$(NC)"
	pipx runpip llm-bawt install fastapi "uvicorn[standard]" httpx
	@echo "$(GREEN)✓ Service deps installed$(NC)"

install-extras-search: check-pipx ## [pipx] Add search deps
	@echo "$(BLUE)Installing search deps...$(NC)"
	pipx runpip llm-bawt install ddgs tavily-python
	@echo "$(GREEN)✓ Search deps installed$(NC)"

uninstall: ## [pipx] Uninstall llm-bawt
	@echo "$(YELLOW)Uninstalling llm-bawt...$(NC)"
	-pipx uninstall llm-bawt 2>$(NULL) || true
	@echo "$(GREEN)✓ llm-bawt uninstalled$(NC)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVELOPMENT (uv-based local .venv)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: dev dev-llama dev-run

dev: check-uv ## [uv] Sync .venv with all extras (mcp, service, search, memory, hf)
	@echo "$(BLUE)Syncing .venv with all extras...$(NC)"
	uv sync --inexact --extra mcp --extra service --extra search --extra memory --extra huggingface
	@echo "$(GREEN)✓ .venv synced$(NC)"
	@echo "  Run: uv run llm --status"

dev-llama: check-uv ## [uv] Add llama-cpp-python to .venv (CUDA-aware)
ifeq ($(WITH_CUDA),true)
	@echo "$(BLUE)Installing llama-cpp-python with CUDA into .venv...$(NC)"
	CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" \
		uv pip install llama-cpp-python --reinstall
else
	@echo "$(BLUE)Installing llama-cpp-python (CPU) into .venv...$(NC)"
	uv pip install llama-cpp-python
endif
	@echo "$(GREEN)✓ llama-cpp-python installed$(NC)"

dev-run: ## [uv] Run llm from .venv (e.g. make dev-run ARGS="--status")
	uv run llm $(ARGS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCKER (works identically on Linux / macOS / Windows)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: run up docker-dev down restart rebuild logs docker-status docker-shell docker-exec

run: down docker-dev logs ## Stop, rebuild dev containers, and tail logs

up: ## Docker compose up (production)
	@echo "Starting containers (production mode)..."
	docker compose up -d
	@echo "$(GREEN)✓ Services started on ports $(MEMORY_PORT) (MCP) and $(SERVICE_PORT) (LLM)$(NC)"

docker-dev: ## Docker compose up (dev mode — mounts ./src)
	@echo "Starting containers (dev mode)..."
	docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_DEV) up -d
	@echo "$(GREEN)✓ Dev mode started — src/ mounted live$(NC)"

down: ## Docker compose down
	docker compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

restart: ## Restart containers
	docker compose restart
	@echo "$(GREEN)✓ Containers restarted$(NC)"

rebuild: ## Rebuild and restart containers
	docker compose down
	docker compose up -d --build
	@echo "$(GREEN)✓ Containers rebuilt and started$(NC)"

logs: ## Follow container logs
	docker compose logs -f --tail=50

docker-status: ## Show container status
	docker compose ps

docker-shell: ## Open bash shell in app container
	docker compose exec app bash

docker-exec: ## Run command in container (make docker-exec CMD="llm --status")
	docker compose exec app $(CMD)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOCAL SERVERS (uv-based — Linux / macOS only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: server-start server-stop server-restart server-status

ifeq ($(DETECTED_OS),Windows)

server-start server-stop server-restart server-status:
	@echo "$(RED)Local server management requires Linux/macOS. Use Docker targets instead (make up / make down).$(NC)"

else

$(RUN_DIR) $(LOG_DIR):
	@$(MKDIR) $@

server-start: check-uv | $(RUN_DIR) $(LOG_DIR) ## Start MCP + LLM services (background)
	@# --- Memory/MCP server ---
	@if [ -f "$(MEMORY_PID)" ] && kill -0 $$(cat "$(MEMORY_PID)") 2>$(NULL); then \
		echo "Memory server already running (pid $$(cat $(MEMORY_PID)))"; \
	else \
		echo "$(BLUE)[mcp] Starting on $(MEMORY_HOST):$(MEMORY_PORT)...$(NC)"; \
		LLM_BAWT_MEMORY_SERVER_VERBOSE= \
		nohup uv run --extra mcp llm-mcp-server --transport http \
			--host $(MEMORY_HOST) --port $(MEMORY_PORT) \
			> $(LOG_DIR)/memory-server.log 2>&1 & \
		echo $$! > $(MEMORY_PID); \
	fi
	@# --- Wait for MCP to be ready ---
	@echo "$(BLUE)[llm] Waiting for MCP server...$(NC)"
	@for i in $$(seq 1 20); do \
		if $(PYTHON) -c "import socket; s=socket.socket(); s.settimeout(0.5); exit(0 if s.connect_ex(('127.0.0.1',$(MEMORY_PORT)))==0 else 1)" 2>$(NULL); then \
			echo "$(GREEN)[llm] MCP server ready$(NC)"; \
			break; \
		fi; \
		sleep 0.5; \
	done
	@# --- LLM service ---
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "llm-service already running (pid $$(cat $(SERVICE_PID)))"; \
	else \
		echo "$(BLUE)[llm] Starting on $(SERVICE_HOST):$(SERVICE_PORT)...$(NC)"; \
		LLM_BAWT_MEMORY_SERVER_URL="$(MEMORY_URL)" \
		nohup uv run --extra service --extra search --extra memory llm-service \
			--host $(SERVICE_HOST) --port $(SERVICE_PORT) \
			> $(LOG_DIR)/llm-service.log 2>&1 & \
		echo $$! > $(SERVICE_PID); \
		echo "$(GREEN)✓ Services started$(NC)"; \
	fi
	@$(MAKE) --no-print-directory server-status

server-stop: ## Stop MCP + LLM services
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "Stopping llm-service (pid $$(cat $(SERVICE_PID)))..."; \
		kill $$(cat "$(SERVICE_PID)") 2>$(NULL) || true; \
		sleep 1; \
		kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL) && kill -9 $$(cat "$(SERVICE_PID)") 2>$(NULL) || true; \
	else \
		echo "llm-service not running"; \
	fi
	@rm -f "$(SERVICE_PID)"
	@if [ -f "$(MEMORY_PID)" ] && kill -0 $$(cat "$(MEMORY_PID)") 2>$(NULL); then \
		echo "Stopping memory server (pid $$(cat $(MEMORY_PID)))..."; \
		kill $$(cat "$(MEMORY_PID)") 2>$(NULL) || true; \
		sleep 1; \
		kill -0 $$(cat "$(MEMORY_PID)") 2>$(NULL) && kill -9 $$(cat "$(MEMORY_PID)") 2>$(NULL) || true; \
	else \
		echo "Memory server not running"; \
	fi
	@rm -f "$(MEMORY_PID)"
	@echo "$(GREEN)✓ Services stopped$(NC)"

server-restart: server-stop server-start ## Restart local services

server-status: ## Show local service status
	@if [ -f "$(MEMORY_PID)" ] && kill -0 $$(cat "$(MEMORY_PID)") 2>$(NULL); then \
		echo "Memory server: $(GREEN)running$(NC) (pid $$(cat $(MEMORY_PID))) on port $(MEMORY_PORT)"; \
	else \
		echo "Memory server: $(RED)stopped$(NC)"; \
	fi
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "llm-service:   $(GREEN)running$(NC) (pid $$(cat $(SERVICE_PID))) on port $(SERVICE_PORT)"; \
	else \
		echo "llm-service:   $(RED)stopped$(NC)"; \
	fi
	@echo "Logs: $(LOG_DIR)/"

endif  # not Windows

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEV TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: lint format typecheck test test-cov

lint: ## Lint with ruff
	uv run ruff check src/

format: ## Format with ruff
	uv run ruff format src/

typecheck: ## Type-check with mypy
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=term-missing

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLEAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: clean clean-all

clean: ## Remove build artifacts and caches
	$(RM_RF) dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache
ifeq ($(DETECTED_OS),Windows)
	-for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>$(NULL) || true
endif
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean ## Remove .venv, logs, and run dirs too
	$(RM_RF) .venv $(LOG_DIR) $(RUN_DIR)
	@echo "$(GREEN)✓ Deep cleaned$(NC)"

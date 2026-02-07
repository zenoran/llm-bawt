#!/usr/bin/env bash
#
# llm-bawt installer
# 
# Usage (from GitHub):
#   curl -fsSL https://raw.githubusercontent.com/zenoran/llm-bawt/master/install.sh | bash
#
# Or with options:
#   curl -fsSL https://raw.githubusercontent.com/zenoran/llm-bawt/master/install.sh | bash -s -- --with-llama --with-hf
#
# Options:
#   --with-llama    Install llama-cpp-python for local GGUF models
#   --with-hf       Install HuggingFace transformers + torch
#   --with-service  Install FastAPI service for background tasks & API
#   --with-search   Install web search providers (DuckDuckGo + Tavily)
#   --all           Install all optional dependencies
#   --no-cuda       Skip CUDA support for llama-cpp-python
#   --uninstall     Remove llm-bawt
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
INSTALL_LLAMA=false
INSTALL_HF=false
INSTALL_SERVICE=false
INSTALL_SEARCH=false
WITH_CUDA=true
UNINSTALL=false
EDITABLE=false
REPO="git+https://github.com/zenoran/llm-bawt.git"
DEV_SYNC=false
FORCE_REBUILD=false
DEPS_ONLY=false  # Install only dependencies, not the project itself (for Docker layer caching)

# Help function
show_help() {
    cat << EOF
llm-bawt installer

USAGE:
    ./install.sh [OPTIONS]

    # From GitHub (one-liner):
    curl -fsSL https://raw.githubusercontent.com/zenoran/llm-bawt/master/install.sh | bash

OPTIONS:
    -h, --help          Show this help message

    Installation:
    --local <PATH>      Install from local path in editable mode (for development)
                        Example: ./install.sh --local .
    --dev               Sync local .venv with all optional deps (for development)
                        Uses uv to install mcp, service, search, llama-cpp (CUDA), HF
    --uninstall         Remove llm-bawt completely

    Optional Dependencies:
    --with-service      Install FastAPI background service (llm-service command)
    --with-llama        Install llama-cpp-python for local GGUF model inference
    --with-hf           Install HuggingFace transformers + torch
    --with-search       Install web search providers (DuckDuckGo + Tavily)
    --all               Install ALL optional dependencies

    Build Options:
    --no-cuda           Skip CUDA/GPU support for llama-cpp-python (CPU only)
    --force-rebuild     Force rebuild of llama-cpp-python (ignore cached build)

EXAMPLES:
    # Basic install from GitHub
    ./install.sh

    # Development install (editable, from current directory)
    ./install.sh --local .

    # Sync local .venv with all deps for development/server.sh
    ./install.sh --dev

    # Full install with all features
    ./install.sh --all

    # Development with service and search
    ./install.sh --local . --with-service --with-search

    # Uninstall
    ./install.sh --uninstall

AFTER INSTALLATION:
    llm --status        Check system status
    llm --list-models   List available models  
    llm "Hello!"        Ask a question
    llm-service         Start background service (if --with-service)

CONFIGURATION:
    ~/.config/llm-bawt/.env          API keys and settings
    ~/.config/llm-bawt/models.yaml   Model definitions

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --with-llama)
            INSTALL_LLAMA=true
            shift
            ;;
        --with-hf)
            INSTALL_HF=true
            shift
            ;;
        --with-service)
            INSTALL_SERVICE=true
            shift
            ;;
        --with-search)
            INSTALL_SEARCH=true
            shift
            ;;
        --all)
            INSTALL_LLAMA=true
            INSTALL_HF=true
            INSTALL_SERVICE=true
            INSTALL_SEARCH=true
            shift
            ;;
        --no-cuda)
            WITH_CUDA=false
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --local)
            # For development: install from local path (editable)
            REPO="$2"
            EDITABLE=true
            shift 2
            ;;
        --dev)
            # Sync local .venv with all optional deps for development
            DEV_SYNC=true
            shift
            ;;
        --deps-only)
            # Install only dependencies, not the project (for Docker layer caching)
            DEPS_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╭─────────────────────────────╮${NC}"
echo -e "${BLUE}│   llm-bawt Installer         │${NC}"
echo -e "${BLUE}╰─────────────────────────────╯${NC}"
echo

# Dev sync - sync local .venv with all optional dependencies
if [ "$DEV_SYNC" = true ]; then
    echo -e "${BLUE}Syncing local .venv with all optional dependencies...${NC}"
    
    # Check for uv
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}✗ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ uv available${NC}"
    
    # Sync all extras (excluding llamacpp - we handle that separately for CUDA)
    # Use --inexact to not remove packages not in the lock file (like our CUDA-built llama-cpp)
    echo -e "${YELLOW}  Syncing extras: mcp, service, search, memory, huggingface...${NC}"
    if [ "$DEPS_ONLY" = true ]; then
        # Docker layer caching: install deps without the project (src/ not available yet)
        # Skip llama-cpp entirely - it will be installed in the next Docker layer
        uv sync --inexact --no-install-project --extra mcp --extra service --extra search --extra memory --extra huggingface
        echo -e "${GREEN}✓ Dependencies synced (deps-only mode)${NC}"
        exit 0
    fi

    uv sync --inexact --extra mcp --extra service --extra search --extra memory --extra huggingface
    echo -e "${GREEN}✓ Base extras synced${NC}"

    # Install llama-cpp-python with CUDA
    echo -e "${BLUE}Installing llama-cpp-python with CUDA...${NC}"
    if [ "$WITH_CUDA" = true ]; then
        if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
            # Check if already installed with CUDA support (skip check if force rebuild)
            LLAMA_CUDA_CHECK=$(uv run python -c "
try:
    import llama_cpp
    # Check if CUDA/cuBLAS support is available
    import llama_cpp.llama_cpp as lib
    if hasattr(lib, 'ggml_cuda_loaded') or 'cuda' in str(dir(lib)).lower() or 'cublas' in str(dir(lib)).lower():
        print('cuda')
    else:
        print('cpu')
except ImportError:
    print('not_installed')
" 2>/dev/null)
            
            if [ "$FORCE_REBUILD" = false ] && [ "$LLAMA_CUDA_CHECK" = "cuda" ]; then
                echo -e "${GREEN}✓ llama-cpp-python already installed with CUDA support${NC}"
            else
                if [ "$LLAMA_CUDA_CHECK" = "not_installed" ]; then
                    echo -e "${YELLOW}  llama-cpp-python not installed, building with CUDA...${NC}"
                elif [ "$LLAMA_CUDA_CHECK" = "cpu" ]; then
                    echo -e "${YELLOW}  llama-cpp-python installed without CUDA, rebuilding...${NC}"
                else
                    echo -e "${YELLOW}  Force rebuilding llama-cpp-python...${NC}"
                fi
                
                CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
                if [ -n "$CUDA_VERSION" ]; then
                    echo -e "${YELLOW}  CUDA version: $CUDA_VERSION${NC}"
                fi
                # Export CMAKE_ARGS for the build process
                # Default to modern CUDA arch list; override with CUDA_ARCHS if needed.
                CUDA_ARCHS="${CUDA_ARCHS:-75;80;86;89;90;120}"
                export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                if [ "$FORCE_REBUILD" = true ]; then
                    uv pip install llama-cpp-python --reinstall --no-cache-dir
                else
                    uv pip install llama-cpp-python --reinstall
                fi
                echo -e "${GREEN}✓ llama-cpp-python installed with CUDA${NC}"
            fi
        else
            echo -e "${YELLOW}  No CUDA detected, installing CPU-only version...${NC}"
            uv pip install llama-cpp-python
            echo -e "${GREEN}✓ llama-cpp-python installed${NC}"
        fi
    else
        echo -e "${YELLOW}  Installing CPU-only version (--no-cuda)...${NC}"
        uv pip install llama-cpp-python
        echo -e "${GREEN}✓ llama-cpp-python installed${NC}"
    fi
    
    echo
    echo -e "${GREEN}╭─────────────────────────────╮${NC}"
    echo -e "${GREEN}│   Dev Sync Complete!        │${NC}"
    echo -e "${GREEN}╰─────────────────────────────╯${NC}"
    echo
    echo -e "Local .venv is now ready for development."
    echo -e "  ${BLUE}./server.sh start${NC}  - Start MCP + LLM service"
    echo -e "  ${BLUE}uv run llm${NC}         - Run llm from local venv"
    echo
    exit 0
fi

# Uninstall
if [ "$UNINSTALL" = true ]; then
    echo -e "${YELLOW}Uninstalling llm-bawt...${NC}"
    pipx uninstall llm-bawt 2>/dev/null || true
    echo -e "${GREEN}✓ llm-bawt uninstalled${NC}"
    exit 0
fi

# Check for Python 3.12+
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.12+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo -e "${RED}✗ Python 3.12+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check/install pipx
echo -e "${BLUE}Checking pipx...${NC}"
if ! command -v pipx &> /dev/null; then
    echo -e "${YELLOW}Installing pipx...${NC}"
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    # Source the updated PATH
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}✓ pipx available${NC}"

# Install llm-bawt
echo -e "${BLUE}Installing llm-bawt...${NC}"

# Build extras string based on flags
# Always include 'memory' extra for sentence-transformers (required for embeddings)
EXTRAS_LIST=("memory")
if [ "$INSTALL_SERVICE" = true ]; then
    EXTRAS_LIST+=("service")
fi
if [ "$INSTALL_SEARCH" = true ]; then
    EXTRAS_LIST+=("search")
fi

# Join extras with commas
EXTRAS=$(IFS=,; echo "${EXTRAS_LIST[*]}")

# Uninstall first if already installed (clean slate for editable)
if pipx list | grep -q "llm-bawt"; then
    echo -e "${YELLOW}  Removing existing installation...${NC}"
    pipx uninstall llm-bawt 2>/dev/null || true
fi

# Build pipx install command
if [ "$EDITABLE" = true ]; then
    echo -e "${YELLOW}  Installing in editable mode from: $REPO${NC}"
    
    # For editable installs, pipx needs the path directly with -e flag
    if [ -n "$EXTRAS" ]; then
        INSTALL_SPEC="${REPO}[${EXTRAS}]"
    else
        INSTALL_SPEC="${REPO}"
    fi
    
    pipx install --editable "$INSTALL_SPEC"
else
    # Non-editable (from git or PyPI)
    if [ -n "$EXTRAS" ]; then
        if [[ "$REPO" == git+* ]]; then
            # Git URL - install base then inject extras
            pipx install --force "$REPO"
        else
            INSTALL_SPEC="${REPO}[${EXTRAS}]"
            pipx install --force "$INSTALL_SPEC"
        fi
    else
        pipx install --force "$REPO"
    fi
fi
echo -e "${GREEN}✓ llm-bawt installed${NC}"

# Install optional dependencies
if [ "$INSTALL_LLAMA" = true ]; then
    echo -e "${BLUE}Installing llama-cpp-python...${NC}"
    
    if [ "$WITH_CUDA" = true ]; then
        # Check for CUDA
        if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
            echo -e "${YELLOW}  CUDA detected, building with GPU support...${NC}"
            # Get CUDA version to determine supported architectures
            # CUDA 12.0 supports up to sm_90 (Hopper), not sm_120 (Blackwell)
            # Use explicit architectures to avoid "native" detection issues
            CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
            if [ -n "$CUDA_VERSION" ]; then
                echo -e "${YELLOW}  CUDA version: $CUDA_VERSION${NC}"
            fi
            # Build for common modern architectures (Turing through Blackwell)
            # sm_75=Turing, sm_80=Ampere, sm_86=Ampere, sm_89=Ada, sm_90=Hopper, sm_120=Blackwell
            CUDA_ARCHS="${CUDA_ARCHS:-75;80;86;89;90;120}" \
            CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
                pipx runpip llm-bawt install llama-cpp-python --force-reinstall --no-cache-dir
        else
            echo -e "${YELLOW}  No CUDA detected, installing CPU-only version...${NC}"
            pipx runpip llm-bawt install llama-cpp-python
        fi
    else
        echo -e "${YELLOW}  Installing CPU-only version (--no-cuda)...${NC}"
        pipx runpip llm-bawt install llama-cpp-python
    fi
    echo -e "${GREEN}✓ llama-cpp-python installed${NC}"
fi

if [ "$INSTALL_HF" = true ]; then
    echo -e "${BLUE}Installing HuggingFace dependencies...${NC}"
    pipx runpip llm-bawt install transformers torch huggingface-hub accelerate
    echo -e "${GREEN}✓ HuggingFace dependencies installed${NC}"
fi

if [ "$INSTALL_SERVICE" = true ]; then
    echo -e "${BLUE}Installing FastAPI service dependencies...${NC}"
    pipx runpip llm-bawt install fastapi "uvicorn[standard]" httpx
    echo -e "${GREEN}✓ FastAPI service dependencies installed${NC}"
fi

if [ "$INSTALL_SEARCH" = true ]; then
    echo -e "${BLUE}Installing web search dependencies...${NC}"
    pipx runpip llm-bawt install ddgs tavily-python
    echo -e "${GREEN}✓ Web search dependencies installed (DuckDuckGo + Tavily)${NC}"
fi

# Verify installation
echo
echo -e "${BLUE}Verifying installation...${NC}"
if command -v llm &> /dev/null; then
    echo -e "${GREEN}✓ 'llm' command available${NC}"
else
    echo -e "${YELLOW}⚠ 'llm' not in PATH. You may need to restart your shell or run:${NC}"
    echo -e "  ${YELLOW}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
fi

if [ "$INSTALL_SERVICE" = true ]; then
    if command -v llm-service &> /dev/null; then
        echo -e "${GREEN}✓ 'llm-service' command available${NC}"
    fi
fi

echo
echo -e "${GREEN}╭─────────────────────────────╮${NC}"
echo -e "${GREEN}│   Installation Complete!    │${NC}"
echo -e "${GREEN}╰─────────────────────────────╯${NC}"
echo
echo -e "Commands available:"
echo -e "  ${BLUE}llm${NC}           - Query LLM models"
echo -e "  ${BLUE}llm-bawt${NC}       - Same as llm"
if [ "$INSTALL_SERVICE" = true ]; then
echo -e "  ${BLUE}llm-service${NC}   - Run background service/API"
fi
echo
echo -e "Quick start:"
echo -e "  ${BLUE}llm --status${NC}           - Check system status"
echo -e "  ${BLUE}llm --list-models${NC}      - List available models"
echo -e "  ${BLUE}llm \"Hello, world!\"${NC}    - Ask a question"
if [ "$INSTALL_SERVICE" = true ]; then
echo
echo -e "Service:"
echo -e "  ${BLUE}llm-service${NC}            - Start the background service"
echo -e "  ${BLUE}llm-service --port 8080${NC} - Start on custom port"
fi
if [ "$INSTALL_SEARCH" = true ]; then
echo
echo -e "Web Search:"
echo -e "  Bots with ${YELLOW}uses_search: true${NC} can now search the internet"
echo -e "  Default: DuckDuckGo (free)"
echo -e "  Set ${YELLOW}LLM_BAWT_TAVILY_API_KEY${NC} in .env for production search"
fi
echo
echo -e "Configuration: ${YELLOW}~/.config/llm-bawt/.env${NC}"
echo

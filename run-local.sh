#!/bin/bash
# ============================================
# Face Recognition API - Local Launcher (uv)
# ============================================
# Quick launcher for local testing using uv package manager
# No Docker required!

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Color functions
write_step() { echo -e "${YELLOW}$1${NC}"; }
write_success() { echo -e "${GREEN}  ✓ $1${NC}"; }
write_error() { echo -e "${RED}  ✗ $1${NC}"; }
write_warning() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
write_info() { echo -e "${CYAN}  $1${NC}"; }

# Parse arguments
INSTALL=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install|-i)
            INSTALL=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

# Display help
if [ "$HELP" = true ]; then
    echo ""
    echo -e "${CYAN}Face Recognition API - Local Launcher${NC}"
    echo -e "${CYAN}=====================================${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "${WHITE}  ./run-local.sh           # Run the API server locally${NC}"
    echo -e "${WHITE}  ./run-local.sh --install # Force install dependencies${NC}"
    echo -e "${WHITE}  ./run-local.sh --help    # Show this help message${NC}"
    echo ""
    exit 0
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}Face Recognition API - Local Launcher${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Step 1: Check if uv is installed
write_step "[1/6] Checking for uv package manager..."
UV_INSTALLED=false

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1)
    write_success "uv is installed: $UV_VERSION"
    UV_INSTALLED=true
else
    UV_INSTALLED=false
fi

if [ "$UV_INSTALLED" = false ]; then
    write_warning "uv is not installed!"
    echo ""
    echo -e "${WHITE}  uv is a fast Python package manager that will be used for this project.${NC}"
    echo ""
    read -p "  Would you like to install uv now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        write_info "Installing uv..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            write_success "uv installed successfully!"
            write_info "Please restart this script to continue."
            echo ""
            exit 0
        else
            write_error "Failed to install uv"
            echo ""
            echo -e "${YELLOW}  Please install uv manually from: https://docs.astral.sh/uv/${NC}"
            exit 1
        fi
    else
        write_error "uv is required to run this project."
        echo ""
        echo -e "${YELLOW}  Install manually with:${NC}"
        echo -e "${WHITE}    curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        echo ""
        exit 1
    fi
fi

# Step 2: Check Python version
write_step "[2/6] Checking Python version..."
# We let uv handle the python versioning, but we check if any python is available
if uv python find &> /dev/null; then
    PYTHON_VERSION=$(uv python find 2>&1)
    write_success "Python found via uv: $PYTHON_VERSION"
else
    write_warning "No default Python found. uv will download it automatically."
fi

# Step 3: Sync dependencies with uv
write_step "[3/6] Syncing dependencies with uv..."
write_info "Ensuring Python 3.12 is available..."
uv python install 3.12 --force  # Ensure we have the compatible version

if [ "$INSTALL" = true ]; then
    write_info "Force installing dependencies..."
    uv sync --reinstall --python 3.12
else
    if [ -d ".venv" ]; then
        write_success "Virtual environment exists"
    fi
    uv sync --python 3.12
fi
write_success "Dependencies synced"

# Step 4: Check configuration
write_step "[4/6] Checking configuration..."
if [ -f ".env" ]; then
    write_success ".env file exists"
else
    write_warning ".env file not found. Creating from template..."
    cp .env.example .env
    write_success ".env created"
    write_warning "Please edit .env with your configuration:"
    write_info "- SUPABASE_URL"
    write_info "- SUPABASE_KEY"
    write_info "- SUPABASE_SERVICE_ROLE_KEY"
    write_info "- ADMIN_SECRET_KEY"
    echo ""
fi

# Step 5: Check model file
write_step "[5/6] Checking ONNX model..."
if [ -f "models/glintr100.onnx" ]; then
    write_success "ArcFace model found"
else
    write_warning "Model file not found!"
    echo ""
    echo -e "${YELLOW}  Download Auraface ONNX model:${NC}"
    echo -e "${WHITE}  1. Visit: https://huggingface.co/fal/AuraFace-v1/tree/main${NC}"
    echo -e "${WHITE}  2. Download: glintr100.onnx (AuraFace r100)${NC}"
    echo -e "${WHITE}  3. Place at: models/glintr100.onnx${NC}"
    echo ""
fi

# Step 6: Check DNN face detector model (auto-download on first run)
write_step "[6/6] Checking face detection model..."
if [ -f "models/deploy.prototxt" ]; then
    write_success "Face detector model cached"
else
    write_info "Face detector will auto-download on first enrollment"
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}Ready to Launch!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Check readiness
ENV_EXISTS=false
MODEL_EXISTS=false

if [ -f ".env" ]; then
    ENV_EXISTS=true
fi

if [ -f "models/glintr100.onnx" ]; then
    MODEL_EXISTS=true
fi

if [ "$MODEL_EXISTS" = false ]; then
    write_warning "Cannot start: ArcFace model is missing"
    echo ""
    echo -e "${YELLOW}  Please download the model first (see instructions above)${NC}"
    echo ""
    exit 1
fi

if [ "$ENV_EXISTS" = false ]; then
    write_warning "Cannot start: .env configuration missing"
    echo ""
    echo -e "${YELLOW}  Please configure .env file first${NC}"
    echo ""
    exit 1
fi

echo -e "${CYAN}Starting Face Recognition API...${NC}"
echo ""
echo -e "${WHITE}  API Server: http://localhost:8000${NC}"
echo -e "${WHITE}  API Docs:   http://localhost:8000/docs${NC}"
echo -e "${WHITE}  Health:     http://localhost:8000/health${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""
echo -e "${CYAN}============================================================${NC}"
echo ""

# Run with uv
uv run python -m src.main

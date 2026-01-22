#!/usr/bin/env bash
# ============================================================================
# Face Recognition API - Local Docker Build Script (Linux/macOS)
# Builds Docker image locally without pushing to registry
# ============================================================================
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🐳 Building Face Recognition API Docker Image (Local)${NC}"
echo -e "${CYAN}=======================================================${NC}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_IMAGE="project-robin"
TAG="latest"
NO_CACHE=""

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-t|--tag <tag>] [--no-cache]"
            echo ""
            echo "Options:"
            echo "  -t, --tag     Docker image tag (default: latest)"
            echo "  --no-cache    Build without using cache"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Usage: $0 [-t|--tag <tag>] [--no-cache]"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${GREEN}📦 Configuration:${NC}"
echo "   Image:    $LOCAL_IMAGE"
echo "   Tag:      $TAG"
echo "   No Cache: ${NO_CACHE:-false}"
echo ""

# Build Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
echo "   This may take a few minutes..."
cd "$PROJECT_ROOT"
docker build $NO_CACHE -t "${LOCAL_IMAGE}:${TAG}" .
echo -e "${GREEN}✅ Docker image built successfully${NC}"

echo ""
echo -e "${CYAN}=======================================================${NC}"
echo -e "${GREEN}✅ Successfully built ${LOCAL_IMAGE}:${TAG}${NC}"
echo ""
echo -e "${CYAN}📋 To run this image:${NC}"
echo ""
echo -e "${YELLOW}   With GPU support (requires NVIDIA Docker):${NC}"
echo "   docker run --gpus all -p 8000:8000 --env-file .env ${LOCAL_IMAGE}:${TAG}"
echo ""
echo -e "${YELLOW}   Without GPU (CPU fallback):${NC}"
echo "   docker run -p 8000:8000 --env-file .env ${LOCAL_IMAGE}:${TAG}"
echo ""
echo -e "${YELLOW}   Using docker-compose:${NC}"
echo "   docker-compose up -d"

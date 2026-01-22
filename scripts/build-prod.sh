#!/usr/bin/env bash
# ============================================================================
# Face Recognition API - Docker Build Script (Linux/macOS)
# Builds and pushes Docker image to GitHub Container Registry (GHCR)
# ============================================================================
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🐳 Building Face Recognition API Docker Image${NC}"
echo -e "${CYAN}================================================${NC}"

# Load .env variables from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOTENV="$PROJECT_ROOT/.env"

if [ -f "$DOTENV" ]; then
    echo -e "${YELLOW}📄 Loading environment variables from .env...${NC}"
    set -a
    # shellcheck disable=SC1090
    . "$DOTENV"
    set +a
fi

# Configuration
REGISTRY=ghcr.io
OWNER=geber-suprabapak
REPO=project-robin
IMAGE=${REGISTRY}/${OWNER}/${REPO}
TAG="latest"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-t|--tag <tag>]"
            echo ""
            echo "Options:"
            echo "  -t, --tag    Docker image tag (default: latest)"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Usage: $0 [-t|--tag <tag>]"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${GREEN}📦 Configuration:${NC}"
echo "   Registry: $REGISTRY"
echo "   Image:    $IMAGE"
echo "   Tag:      $TAG"
echo ""

# Check required env vars
if [ -z "$GHCR_USERNAME" ] || [ -z "$GHCR_TOKEN" ]; then
    echo -e "${RED}❌ Error: Please set GHCR_USERNAME and GHCR_TOKEN environment variables.${NC}"
    echo ""
    echo -e "${YELLOW}You can set these in your .env file:${NC}"
    echo "   GHCR_USERNAME=your-github-username"
    echo "   GHCR_TOKEN=your-github-personal-access-token"
    exit 1
fi

# Login to GHCR
echo -e "${YELLOW}🔐 Logging in to GitHub Container Registry...${NC}"
echo "$GHCR_TOKEN" | docker login $REGISTRY -u "$GHCR_USERNAME" --password-stdin
echo -e "${GREEN}✅ Successfully logged in to GHCR${NC}"

# Build Docker image
echo ""
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
echo "   This may take a few minutes..."
cd "$PROJECT_ROOT"
docker build -t "${IMAGE}:${TAG}" .
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Push to GHCR
echo ""
echo -e "${YELLOW}🚀 Pushing image to GHCR...${NC}"
docker push "${IMAGE}:${TAG}"

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${GREEN}✅ Successfully pushed ${IMAGE}:${TAG}${NC}"
echo ""
echo -e "${CYAN}📋 To pull and run this image:${NC}"
echo "   docker pull ${IMAGE}:${TAG}"
echo "   docker run --gpus all -p 8000:8000 ${IMAGE}:${TAG}"

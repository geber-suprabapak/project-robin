#!/bin/bash

# ============================================================================
# Build and push CPU-only Docker image to GHCR.
# uses Dockerfile.cpu and tags with cpu- prefix.
# ============================================================================

# Default values
TAG="cpu-latest"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) TAG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo -e "\033[0;36m🐳 Building CPU Docker Image\033[0m"

# Load .env with improved parsing
if [ -f .env ]; then
    echo -e "\033[0;33m📄 Loading environment variables from .env...\033[0m"
    while IFS='=' read -r name value || [ -n "$name" ]; do
        # Skip comments and empty lines
        [[ "$name" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${name// }" ]] && continue
        
        # Trim whitespace and quotes
        name=$(echo "$name" | xargs)
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^['\"]//" -e "s/['\"]$//")
        
        export "$name=$value"
    done < .env
fi


REGISTRY="ghcr.io"
OWNER="geber-suprabapak"
REPO="project-robin"
IMAGE="$REGISTRY/$OWNER/$REPO"

if [ -z "$GHCR_USERNAME" ] || [ -z "$GHCR_TOKEN" ]; then
    echo "❌ Missing GHCR_USERNAME or GHCR_TOKEN"
    exit 1
fi

echo "$GHCR_TOKEN" | docker login $REGISTRY -u "$GHCR_USERNAME" --password-stdin

echo "🔨 Building $IMAGE:$TAG (CPU)..."
docker build -t "$IMAGE:$TAG" -f Dockerfile.cpu .
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "🚀 Pushing to GHCR..."
docker push "$IMAGE:$TAG"
if [ $? -ne 0 ]; then
    echo "❌ Push failed"
    exit 1
fi

echo -e "\033[0;32m✅ Done: $IMAGE:$TAG\033[0m"

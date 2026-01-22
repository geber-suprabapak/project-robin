#!/bin/bash

# ============================================================================
# Build GPU Docker image locally (no push).
# ============================================================================

# Default values
TAG="latest"
NO_CACHE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) TAG="$2"; shift ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo -e "\033[0;36m🐳 Building GPU Docker Image (Local)\033[0m"
LOCAL_IMAGE="project-robin"

BUILD_ARGS=(-t "$LOCAL_IMAGE:$TAG" -f Dockerfile .)
if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS=(--no-cache "${BUILD_ARGS[@]}")
fi

docker build "${BUILD_ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo -e "\033[0;32m✅ Built $LOCAL_IMAGE:$TAG\033[0m"
echo -e "Run with: \033[0;33mdocker run --gpus all -p 8000:8000 --env-file .env $LOCAL_IMAGE:$TAG\033[0m"

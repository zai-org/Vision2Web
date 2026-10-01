#!/bin/bash

# Build script for Vision2Web sandbox Docker image

set -e

echo "Building Vision2Web sandbox Docker image..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build the image
docker build \
    -t vision2web-sandbox:latest \
    -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

echo "✓ Docker image built successfully: vision2web-sandbox:latest"
echo ""
echo "You can now run Vision2Web with:"
echo "  vision2web inference --sandbox vision2web-sandbox:latest ..."

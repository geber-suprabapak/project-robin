#!/bin/bash
# ============================================
# Face Recognition API - Setup Script
# ============================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}Face Recognition API - Setup Script${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    if [[ $PYTHON_VERSION =~ Python\ 3\.([0-9]+) ]]; then
        MINOR_VERSION=${BASH_REMATCH[1]}
        if [ "$MINOR_VERSION" -ge 11 ]; then
            echo -e "${GREEN}  ✓ Python version OK: $PYTHON_VERSION${NC}"
        else
            echo -e "${RED}  ✗ Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
            exit 1
        fi
    else
        echo -e "${RED}  ✗ Could not determine Python version${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ✗ Python not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}[3/6] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "${YELLOW}[4/6] Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# Check environment file
echo -e "${YELLOW}[5/6] Checking configuration...${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}  ✓ .env file exists${NC}"
else
    echo -e "${YELLOW}  ⚠ .env file not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}  ✓ .env created. Please edit with your Supabase credentials${NC}"
fi

# Check model file
echo -e "${YELLOW}[6/7] Checking model file...${NC}"
if [ -f "models/arcface_r100_224x224.onnx" ]; then
    echo -e "${GREEN}  ✓ Model file found${NC}"
else
    echo ""
    echo -e "${RED}  ⚠ WARNING: Model file not found!${NC}"
    echo ""
    echo -e "${YELLOW}  You need to download an ArcFace ONNX model:${NC}"
    echo -e "${WHITE}  1. Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo${NC}"
    echo -e "${WHITE}  2. Download: ArcFace ResNet100 (Hi-Res 224x224)${NC}"
    echo -e "${WHITE}  3. Place at: models/arcface_r100_224x224.onnx${NC}"
    echo ""
    echo -e "${YELLOW}  Model Requirements:${NC}"
    echo -e "${WHITE}    - File type: .onnx${NC}"
    echo -e "${WHITE}    - Input shape: (1, 3, 224, 224)${NC}"
    echo -e "${WHITE}    - Output shape: (1, 512)${NC}"
    echo -e "${WHITE}    - Size: ~200-500 MB${NC}"
    echo ""
fi

# Check/Download anti-spoofing models
echo -e "${YELLOW}[7/7] Checking anti-spoofing models...${NC}"
ANTI_SPOOF_MODEL_SCALE27="models/anti_spoof_2.7_80x80.onnx"
ANTI_SPOOF_MODEL_SCALE40="models/anti_spoof_4.0_80x80.onnx"
DOWNLOAD_BASE_URL="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"

if [ -f "$ANTI_SPOOF_MODEL_SCALE27" ] && [ -f "$ANTI_SPOOF_MODEL_SCALE40" ]; then
    echo -e "${GREEN}  ✓ Anti-spoofing models found${NC}"
else
    echo -e "${YELLOW}  ⚠ Anti-spoofing models not found. Downloading...${NC}"
    
    # Create models directory if not exists
    mkdir -p models
    
    # Download MiniFASNetV2 model
    SOURCE_MODEL="MiniFASNetV2.onnx"
    TEMP_PATH="models/$SOURCE_MODEL"
    
    echo -e "${CYAN}  Downloading MiniFASNetV2.onnx...${NC}"
    if curl -L -o "$TEMP_PATH" "$DOWNLOAD_BASE_URL/$SOURCE_MODEL" 2>/dev/null || wget -O "$TEMP_PATH" "$DOWNLOAD_BASE_URL/$SOURCE_MODEL" 2>/dev/null; then
        # Copy to both scale paths (same model, different scales handled in code)
        cp "$TEMP_PATH" "$ANTI_SPOOF_MODEL_SCALE27"
        cp "$TEMP_PATH" "$ANTI_SPOOF_MODEL_SCALE40"
        rm "$TEMP_PATH"
        
        echo -e "${GREEN}  ✓ Anti-spoofing models downloaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to download anti-spoofing models${NC}"
        echo -e "${YELLOW}  Please download manually from: $DOWNLOAD_BASE_URL${NC}"
    fi
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Check if everything is ready
MODEL_EXISTS=false
ENV_EXISTS=false
ANTI_SPOOF_EXISTS=false

if [ -f "models/arcface_r100_224x224.onnx" ]; then
    MODEL_EXISTS=true
fi

if [ -f ".env" ]; then
    ENV_EXISTS=true
fi

if [ -f "$ANTI_SPOOF_MODEL_SCALE27" ] && [ -f "$ANTI_SPOOF_MODEL_SCALE40" ]; then
    ANTI_SPOOF_EXISTS=true
fi

if [ "$MODEL_EXISTS" = true ] && [ "$ENV_EXISTS" = true ] && [ "$ANTI_SPOOF_EXISTS" = true ]; then
    echo -e "${GREEN}✅ All requirements met! Ready to start.${NC}"
    echo ""
    echo -e "${CYAN}To start the API server:${NC}"
    echo -e "${WHITE}  python main.py${NC}"
    echo ""
    echo -e "${CYAN}Or with Docker:${NC}"
    echo -e "${WHITE}  docker-compose up -d${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Please complete the following:${NC}"
    echo ""
    if [ "$MODEL_EXISTS" = false ]; then
        echo -e "${RED}  [ ] Download ArcFace ONNX model to models/${NC}"
    fi
    if [ "$ENV_EXISTS" = false ]; then
        echo -e "${RED}  [ ] Configure .env with Supabase credentials${NC}"
    fi
    if [ "$ANTI_SPOOF_EXISTS" = false ]; then
        echo -e "${RED}  [ ] Download anti-spoofing models to models/${NC}"
    fi
    echo ""
    echo -e "${CYAN}See README.md for detailed instructions.${NC}"
    echo ""
fi

echo -e "${CYAN}Documentation:${NC}"
echo -e "${WHITE}  - README.md - Full setup guide${NC}"
echo -e "${WHITE}  - API Docs: http://localhost:8000/docs (after starting)${NC}"
echo ""


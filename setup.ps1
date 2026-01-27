# ============================================
# Face Recognition API - Quick Start Script
# ============================================

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Face Recognition API - Setup Script" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.([0-9]+)") {
    $minorVersion = [int]$Matches[1]
    if ($minorVersion -ge 11) {
        Write-Host "  ✓ Python version OK: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Python 3.11+ required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ✗ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# Check environment file
Write-Host "[5/6] Checking configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "  ⚠ .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "  ✓ .env created. Please edit with your Supabase credentials" -ForegroundColor Yellow
}

# Check model file
Write-Host "[6/7] Checking model file..." -ForegroundColor Yellow
if (Test-Path "models/arcface_r100_224x224.onnx") {
    Write-Host "  ✓ Model file found" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ⚠ WARNING: Model file not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  You need to download an ArcFace ONNX model:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://github.com/deepinsight/insightface/tree/master/model_zoo" -ForegroundColor White
    Write-Host "  2. Download: ArcFace ResNet100 (Hi-Res 224x224)" -ForegroundColor White
    Write-Host "  3. Place at: models/arcface_r100_224x224.onnx" -ForegroundColor White
    Write-Host ""
    Write-Host "  Model Requirements:" -ForegroundColor Yellow
    Write-Host "    - File type: .onnx" -ForegroundColor White
    Write-Host "    - Input shape: (1, 3, 224, 224)" -ForegroundColor White
    Write-Host "    - Output shape: (1, 512)" -ForegroundColor White
    Write-Host "    - Size: ~200-500 MB" -ForegroundColor White
    Write-Host ""
}

# Check/Download anti-spoofing models
Write-Host "[7/7] Checking anti-spoofing models..." -ForegroundColor Yellow
$antiSpoofModelScale27 = "models/anti_spoof_2.7_80x80.onnx"
$antiSpoofModelScale40 = "models/anti_spoof_4.0_80x80.onnx"
$downloadBaseUrl = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"

if ((Test-Path $antiSpoofModelScale27) -and (Test-Path $antiSpoofModelScale40)) {
    Write-Host "  ✓ Anti-spoofing models found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Anti-spoofing models not found. Downloading..." -ForegroundColor Yellow
    
    # Create models directory if not exists
    if (-not (Test-Path "models")) {
        New-Item -ItemType Directory -Path "models" | Out-Null
    }
    
    # Download MiniFASNetV2 model (used for both scales)
    $sourceModel = "MiniFASNetV2.onnx"
    $tempPath = "models/$sourceModel"
    
    try {
        Write-Host "  Downloading MiniFASNetV2.onnx..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri "$downloadBaseUrl/$sourceModel" -OutFile $tempPath -UseBasicParsing
        
        # Copy to both scale paths (same model, different scales handled in code)
        Copy-Item $tempPath $antiSpoofModelScale27
        Copy-Item $tempPath $antiSpoofModelScale40
        Remove-Item $tempPath
        
        Write-Host "  ✓ Anti-spoofing models downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to download anti-spoofing models: $_" -ForegroundColor Red
        Write-Host "  Please download manually from: $downloadBaseUrl" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if everything is ready
$modelExists = Test-Path "models/arcface_r100_224x224.onnx"
$envExists = Test-Path ".env"
$antiSpoofExists = (Test-Path $antiSpoofModelScale27) -and (Test-Path $antiSpoofModelScale40)

if ($modelExists -and $envExists -and $antiSpoofExists) {
    Write-Host "✅ All requirements met! Ready to start." -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the API server:" -ForegroundColor Cyan
    Write-Host "  python main.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Or with Docker:" -ForegroundColor Cyan
    Write-Host "  docker-compose up -d" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "⚠️  Please complete the following:" -ForegroundColor Yellow
    Write-Host ""
    if (-not $modelExists) {
        Write-Host "  [ ] Download ArcFace ONNX model to models/" -ForegroundColor Red
    }
    if (-not $envExists) {
        Write-Host "  [ ] Configure .env with Supabase credentials" -ForegroundColor Red
    }
    if (-not $antiSpoofExists) {
        Write-Host "  [ ] Download anti-spoofing models to models/" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "See README.md for detailed instructions." -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - README.md - Full setup guide" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/docs (after starting)" -ForegroundColor White
Write-Host ""


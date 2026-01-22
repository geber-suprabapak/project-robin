# ============================================
# Face Recognition API - Local Launcher (uv)
# ============================================
# Quick launcher for local testing using uv package manager
# No Docker required!

param(
    [switch]$Install,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Color functions
function Write-Step { param($msg) Write-Host $msg -ForegroundColor Yellow }
function Write-Success { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "  $msg" -ForegroundColor Cyan }

# Display help
if ($Help) {
    Write-Host ""
    Write-Host "Face Recognition API - Local Launcher" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run-local.ps1           # Run the API server locally" -ForegroundColor White
    Write-Host "  .\run-local.ps1 -Install  # Force install dependencies" -ForegroundColor White
    Write-Host "  .\run-local.ps1 -Help     # Show this help message" -ForegroundColor White
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Face Recognition API - Local Launcher" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if uv is installed
Write-Step "[1/6] Checking for uv package manager..."
$uvInstalled = $false
try {
    $uvVersion = uv --version 2>&1
    if ($?) {
        Write-Success "uv is installed: $uvVersion"
        $uvInstalled = $true
    }
} catch {
    $uvInstalled = $false
}

if (-not $uvInstalled) {
    Write-Warning "uv is not installed!"
    Write-Host ""
    Write-Host "  uv is a fast Python package manager that will be used for this project." -ForegroundColor White
    Write-Host ""
    $response = Read-Host "  Would you like to install uv now? (y/n)"
    
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Info "Installing uv..."
        try {
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            Write-Success "uv installed successfully!"
            Write-Info "Please restart this script to continue."
            Write-Host ""
            exit 0
        } catch {
            Write-Error "Failed to install uv: $_"
            Write-Host ""
            Write-Host "  Please install uv manually from: https://docs.astral.sh/uv/" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Error "uv is required to run this project."
        Write-Host ""
        Write-Host "  Install manually with:" -ForegroundColor Yellow
        Write-Host "    powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'" -ForegroundColor White
        Write-Host ""
        exit 1
    }
}

# Step 2: Check Python version
Write-Step "[2/6] Checking Python version..."
# We let uv handle the python versioning, but we check if any python is available
try {
    $pythonVersion = uv python find 2>&1
    Write-Success "Python found via uv: $pythonVersion"
} catch {
    Write-Warning "No default Python found. uv will download it automatically."
}

# Step 3: Sync dependencies with uv
Write-Step "[3/6] Syncing dependencies with uv..."
Write-Info "Ensuring Python 3.12 is available..."
uv python install 3.12 --force  # Ensure we have the compatible version

if ($Install) {
    Write-Info "Force installing dependencies..."
    uv sync --reinstall --python 3.12
} else {
    if (Test-Path ".venv") {
        Write-Success "Virtual environment exists"
    }
    uv sync --python 3.12
}
Write-Success "Dependencies synced"

# Step 4: Check configuration
Write-Step "[4/6] Checking configuration..."
if (Test-Path ".env") {
    Write-Success ".env file exists"
} else {
    Write-Warning ".env file not found. Creating from template..."
    Copy-Item .env.example .env
    Write-Success ".env created"
    Write-Warning "Please edit .env with your configuration:"
    Write-Info "- SUPABASE_URL"
    Write-Info "- SUPABASE_KEY"
    Write-Info "- SUPABASE_SERVICE_ROLE_KEY"
    Write-Info "- ADMIN_SECRET_KEY"
    Write-Host ""
}

# Step 5: Check model file
Write-Step "[5/6] Checking ONNX model..."
if (Test-Path "models/glintr100.onnx") {
    Write-Success "ArcFace model found"
} else {
    Write-Warning "Model file not found!"
    Write-Host ""
    Write-Host "  Download Auraface ONNX model:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://huggingface.co/fal/AuraFace-v1/tree/main" -ForegroundColor White
    Write-Host "  2. Download: glintr100.onnx (AuraFace r100)" -ForegroundColor White
    Write-Host "  3. Place at: models/glintr100.onnx" -ForegroundColor White
    Write-Host ""
}

# Step 6: Check DNN face detector model (auto-download on first run)
Write-Step "[6/6] Checking face detection model..."
if (Test-Path "models/deploy.prototxt") {
    Write-Success "Face detector model cached"
} else {
    Write-Info "Face detector will auto-download on first enrollment"
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Ready to Launch!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check readiness
$envExists = Test-Path ".env"
$modelExists = Test-Path "models/glintr100.onnx"

if (-not $modelExists) {
    Write-Warning "Cannot start: ArcFace model is missing"
    Write-Host ""
    Write-Host "  Please download the model first (see instructions above)" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

if (-not $envExists) {
    Write-Warning "Cannot start: .env configuration missing"
    Write-Host ""
    Write-Host "  Please configure .env file first" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "Starting Face Recognition API..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  API Server: http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health:     http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Run with uv
uv run python -m src.main

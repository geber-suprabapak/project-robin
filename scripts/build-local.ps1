<#
.SYNOPSIS
    Build Docker image locally without pushing to registry.

.DESCRIPTION
    This script builds the Face Recognition API Docker image locally for testing.
    No registry credentials required.

.PARAMETER Tag
    Optional image tag (default: latest)

.PARAMETER NoCache
    Build without using cache

.EXAMPLE
    .\scripts\build-local.ps1
    .\scripts\build-local.ps1 -Tag "dev"
    .\scripts\build-local.ps1 -NoCache
#>
param(
    [string]$Tag = 'latest',
    [switch]$NoCache
)

# Determine project root and switch directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "🐳 Building Face Recognition API Docker Image (Local)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Configuration
$LocalImage = 'project-robin'

Write-Host ""
Write-Host "📦 Configuration:" -ForegroundColor Green
Write-Host "   Image:    $LocalImage"
Write-Host "   Tag:      $Tag"
Write-Host "   No Cache: $NoCache"
Write-Host ""

# Build Docker image
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes..."

$buildArgs = @('-t', "$($LocalImage):$Tag", '.')
if ($NoCache) {
    $buildArgs = @('--no-cache') + $buildArgs
}

docker build @buildArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker build failed"
    exit 1
}

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "✅ Successfully built $($LocalImage):$Tag" -ForegroundColor Green
Write-Host ""
Write-Host "📋 To run this image:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   With GPU support (requires NVIDIA Docker):" -ForegroundColor Yellow
Write-Host "   docker run --gpus all -p 8000:8000 --env-file .env $($LocalImage):$Tag"
Write-Host ""
Write-Host "   Without GPU (CPU fallback):" -ForegroundColor Yellow
Write-Host "   docker run -p 8000:8000 --env-file .env $($LocalImage):$Tag"
Write-Host ""
Write-Host "   Using docker-compose:" -ForegroundColor Yellow
Write-Host "   docker-compose up -d"

<#
.SYNOPSIS
    Build and push Docker image to GitHub Container Registry (GHCR) on Windows PowerShell.

.DESCRIPTION
    This script builds the Face Recognition API Docker image and pushes it to GHCR.
    Requires GHCR_USERNAME and GHCR_TOKEN environment variables (can be set in .env file).

.PARAMETER Tag
    Optional image tag (default: latest)

.EXAMPLE
    .\scripts\build-prod.ps1
    .\scripts\build-prod.ps1 -Tag "v1.0.0"
#>
param(
    [string]$Tag = 'latest'
)

# Determine project root and switch directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "🐳 Building Face Recognition API Docker Image" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Load .env from project root
$envFile = Join-Path $projectRoot '.env'
if (Test-Path $envFile) {
    Write-Host "📄 Loading environment variables from .env..." -ForegroundColor Yellow
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*$') { return }
        $parts = $_ -split '=', 2
        if ($parts.Length -eq 2) {
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
        }
    }
}

# Configuration
$Registry = 'ghcr.io'
$Owner = 'geber-suprabapak'
$Repo = 'project-robin'
$Image = "$Registry/$Owner/$Repo"

Write-Host ""
Write-Host "📦 Configuration:" -ForegroundColor Green
Write-Host "   Registry: $Registry"
Write-Host "   Image:    $Image"
Write-Host "   Tag:      $Tag"
Write-Host ""

# Check credentials
if (-not $Env:GHCR_USERNAME -or -not $Env:GHCR_TOKEN) {
    Write-Error "❌ Please set GHCR_USERNAME and GHCR_TOKEN environment variables."
    Write-Host ""
    Write-Host "You can set these in your .env file:" -ForegroundColor Yellow
    Write-Host "   GHCR_USERNAME=your-github-username"
    Write-Host "   GHCR_TOKEN=your-github-personal-access-token"
    exit 1
}

# Login to GHCR
Write-Host "🔐 Logging in to GitHub Container Registry..." -ForegroundColor Yellow
$loginResult = echo $Env:GHCR_TOKEN | docker login $Registry -u $Env:GHCR_USERNAME --password-stdin 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to login to GHCR: $loginResult"
    exit 1
}
Write-Host "✅ Successfully logged in to GHCR" -ForegroundColor Green

# Build Docker image
Write-Host ""
Write-Host "🔨 Building Docker image..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes..."
docker build -t "$($Image):$Tag" .
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker build failed"
    exit 1
}
Write-Host "✅ Docker image built successfully" -ForegroundColor Green

# Push to GHCR
Write-Host ""
Write-Host "🚀 Pushing image to GHCR..." -ForegroundColor Yellow
docker push "$($Image):$Tag"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker push failed"
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Successfully pushed $($Image):$Tag" -ForegroundColor Green
Write-Host ""
Write-Host "📋 To pull and run this image:" -ForegroundColor Cyan
Write-Host "   docker pull $($Image):$Tag"
Write-Host "   docker run --gpus all -p 8000:8000 $($Image):$Tag"

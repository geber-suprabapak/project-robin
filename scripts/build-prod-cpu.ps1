<#
.SYNOPSIS
    Build and push CPU-only Docker image to GHCR.

.DESCRIPTION
    Builds using Dockerfile.cpu and pushes to GHCR with cpu- prefix/tag.
    Default tag is 'cpu-latest'.
#>
param(
    [string]$Tag = 'cpu-latest'
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location -Path $projectRoot

Write-Host "🐳 Building CPU Docker Image" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Load .env with improved parsing (handles quotes and spaces)
$envFile = Join-Path $projectRoot '.env'
if (Test-Path $envFile) {
    Write-Host "📄 Loading environment variables from .env..." -ForegroundColor Yellow
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#\s][^=]*)\s*=\s*(.*)$') {
            $name = $Matches[1].Trim()
            $value = $Matches[2].Trim().Trim("'").Trim('"')
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Configuration
$Registry = 'ghcr.io'
$Owner = 'geber-suprabapak'
$Repo = 'project-robin'
$Image = "$Registry/$Owner/$Repo"

# Validation
if ([string]::IsNullOrWhiteSpace($Env:GHCR_USERNAME) -or [string]::IsNullOrWhiteSpace($Env:GHCR_TOKEN)) {
    Write-Error "❌ GHCR_USERNAME or GHCR_TOKEN is not set. Check your .env file."
    exit 1
}

# Login
Write-Host "🔐 Logging in to GHCR..." -ForegroundColor Yellow
$Env:GHCR_TOKEN | docker login $Registry -u $Env:GHCR_USERNAME --password-stdin
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Login failed"
    exit 1
}

# Build
Write-Host "🔨 Building $($Image):$Tag (CPU)..." -ForegroundColor Yellow
docker build -t "$($Image):$Tag" -f Dockerfile.cpu .
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker build failed"
    exit 1
}

# Push
Write-Host "🚀 Pushing image to GHCR..." -ForegroundColor Yellow
docker push "$($Image):$Tag"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Docker push failed"
    exit 1
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Successfully pushed $($Image):$Tag" -ForegroundColor Green


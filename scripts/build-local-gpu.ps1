<#
.SYNOPSIS
    Build GPU Docker image locally (no push).
#>
param(
    [string]$Tag = 'latest',
    [switch]$NoCache
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "🐳 Building GPU Docker Image (Local)" -ForegroundColor Cyan
$LocalImage = 'project-robin'

$buildArgs = @('-t', "$($LocalImage):$Tag", '-f', 'Dockerfile', '.')
if ($NoCache) { $buildArgs = @('--no-cache') + $buildArgs }

docker build @buildArgs
if ($LASTEXITCODE -ne 0) { 
    Write-Error "❌ Build failed"
    exit 1 
}

Write-Host "✅ Built $($LocalImage):$Tag" -ForegroundColor Green
Write-Host "Run with: docker run --gpus all -p 8000:8000 --env-file .env $($LocalImage):$Tag"

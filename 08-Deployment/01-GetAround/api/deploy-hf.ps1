#Requires -Version 5.1
<#
.SYNOPSIS
  Synchronise api/ (monorepo local) -> clone du Space HF ZerphirosX/getaround-api, puis commit + push.
.PARAMETER SpacePath
  Chemin vers le clone local du Space HF.
.PARAMETER Message
  Message de commit (défaut : horodatage).
.EXAMPLE
  .\deploy-hf.ps1 -SpacePath "c:\Users\capit\Dev\Perso\getaround-api" -Message "doc: Swagger enrichi"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SpacePath,

    [string]$Message = ("Sync from monorepo {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"))
)

$ErrorActionPreference = "Stop"

$ApiDir = $PSScriptRoot
if (-not (Test-Path $ApiDir)) { throw "Dossier api/ introuvable : $ApiDir" }
if (-not (Test-Path $SpacePath)) { throw "Dossier Space introuvable : $SpacePath" }
if (-not (Test-Path (Join-Path $SpacePath ".git"))) {
    throw "$SpacePath n'est pas un repo git. Clonez d'abord : git clone https://huggingface.co/spaces/ZerphirosX/getaround-api"
}

$filesToSync = @("app.py", "Dockerfile", "requirements.txt")

foreach ($f in $filesToSync) {
    $src = Join-Path $ApiDir $f
    $dst = Join-Path $SpacePath $f
    if (-not (Test-Path $src)) { throw "Fichier source manquant : $src" }
    Copy-Item -Force $src $dst
    Write-Host "Copié : $f" -ForegroundColor Green
}

Push-Location $SpacePath
try {
    git add $filesToSync
    $staged = git diff --cached --name-only
    if (-not $staged) {
        Write-Host "Aucun changement à committer." -ForegroundColor Yellow
        return
    }
    git commit -m $Message
    git push
    Write-Host "Déployé sur HF. Suivez le build :" -ForegroundColor Cyan
    Write-Host "  https://huggingface.co/spaces/ZerphirosX/getaround-api" -ForegroundColor Cyan
} finally {
    Pop-Location
}

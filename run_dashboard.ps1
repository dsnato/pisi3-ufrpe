# Script para executar o Dashboard
# Uso: .\run_dashboard.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   HOTEL BOOKING ANALYSIS DASHBOARD" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se venv existe
if (-not (Test-Path "venv")) {
    Write-Host "❌ Ambiente virtual não encontrado!" -ForegroundColor Red
    Write-Host "   Execute: py -m venv venv" -ForegroundColor Yellow
    Write-Host "   Depois: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "   E: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Ativar venv
Write-Host "🔧 Ativando ambiente virtual..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Verificar se as dependências estão instaladas
Write-Host "📦 Verificando dependências..." -ForegroundColor Green
$dash = pip show dash 2>$null
if (-not $dash) {
    Write-Host "❌ Dependências não instaladas!" -ForegroundColor Red
    Write-Host "   Execute: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Ambiente configurado!" -ForegroundColor Green
Write-Host ""

# Executar dashboard
Write-Host "🚀 Iniciando Dashboard..." -ForegroundColor Cyan
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Acesse em: http://127.0.0.1:8050" -ForegroundColor Yellow
Write-Host "   Ou: http://localhost:8050" -ForegroundColor Yellow
Write-Host "   Pressione Ctrl+C para parar" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Executar o dashboard
python DASH/dash_interativo_ml.py

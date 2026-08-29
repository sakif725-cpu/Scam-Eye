# AI Fraud Detection System - Windows PowerShell Setup Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 🛡️ AI Fraud Call and Message Detector - Setup Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Host "❌ Error: Python 3.9+ was not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://python.org and add it to PATH." -ForegroundColor Yellow
    exit 1
}

Write-Host "✔️ Found Python: $($pythonCmd.Source)" -ForegroundColor Green

# 2. Setup Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment (.venv)..." -ForegroundColor Yellow
    & python -m venv .venv
} else {
    Write-Host "✔️ Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 3. Activate Virtual Environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

# 4. Install Dependencies
Write-Host "📥 Installing / Updating dependencies from requirements.txt..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 5. Dataset Preprocessing & Model Check
Write-Host "⚙️ Validating Dataset and AI Models..." -ForegroundColor Yellow
if (-not (Test-Path "data/processed/train.csv")) {
    Write-Host "📊 Generating processed dataset splits..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe src/data_preprocessor.py
}

if (-not (Test-Path "models/baseline/logistic_regression_model.joblib")) {
    Write-Host "🧠 Training Baseline ML Models (TF-IDF + Logistic Regression / Naive Bayes)..." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe src/train_baseline.py
    & .\.venv\Scripts\python.exe src/evaluate_models.py
    & .\.venv\Scripts\python.exe src/model_selector.py
}

# 6. Run Unit Tests
Write-Host "🧪 Running test suite..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pytest tests/ -v

Write-Host "============================================================" -ForegroundColor Green
Write-Host " 🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To launch the Streamlit Web Application:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "To launch the FastAPI REST API Server:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python main.py" -ForegroundColor White
Write-Host "  or: .\.venv\Scripts\uvicorn main:app --reload --port 8000" -ForegroundColor White
Write-Host ""


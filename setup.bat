@echo off
REM AI Fraud Detection System - Windows Batch Setup Script
echo ============================================================
echo  🛡️ AI Fraud Call and Message Detector - Setup Script
echo ============================================================

REM 1. Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM 2. Create Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [INFO] Virtual environment (.venv) already exists.
)

REM 3. Activate Virtual Environment & Install Dependencies
echo [INFO] Installing/Updating dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. Check Dataset & Baseline Model
if not exist "data\processed\train.csv" (
    echo [INFO] Generating dataset splits...
    python src\data_preprocessor.py
)

if not exist "models\baseline\logistic_regression_model.joblib" (
    echo [INFO] Training baseline ML models...
    python src\train_baseline.py
    python src\evaluate_models.py
    python src\model_selector.py
)

REM 5. Run Unit Tests
echo [INFO] Running test suite...
pytest tests\ -v

echo ============================================================
echo  🎉 Setup completed successfully!
echo ============================================================
echo To run the Streamlit Web App:
echo   .venv\Scripts\streamlit run app.py
echo.
echo To run the FastAPI REST API:
echo   .venv\Scripts\python main.py
echo ============================================================
pause


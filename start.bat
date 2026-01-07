@echo off
cd /d "%~dp0"
echo Starting RAG Parameter Tuning Tool...
echo.
echo Installing dependencies (first time only)...
pip install -r requirements.txt
echo.
echo Starting server...
python app.py
pause


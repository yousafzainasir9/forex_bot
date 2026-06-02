@echo off
REM Start the web dashboard at http://127.0.0.1:5000
cd /d "%~dp0"
uv run python -m webapp.app
pause

@echo off
REM Start the trading bot (DEMO). Requires .env configured and MT5 running.
cd /d "%~dp0"
uv run python main.py
pause

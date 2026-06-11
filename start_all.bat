@echo off
rem One-click launcher: dashboard in the background, bot in this window.
rem Dashboard: http://localhost:8765 (phone URL printed in dashboard.log)
cd /d "%~dp0"
start "kronos-dashboard" /min cmd /c "python dashboard_server.py > dashboard.log 2>&1"
python main.py

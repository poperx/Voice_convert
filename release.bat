@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%~dp0python;%~dp0python\DLLs;%PATH%"
"%~dp0python\python.exe" "%~dp0fix_env.py"
if errorlevel 1 (
  echo 部署没有完成。
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0voice_gui.py"

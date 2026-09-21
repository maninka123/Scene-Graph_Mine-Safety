@echo off
setlocal
cd /d "%~dp0"
title MineGraph Studio

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_app.ps1"

if errorlevel 1 (
  echo.
  echo MineGraph Studio could not start. Review the message above.
  pause
)


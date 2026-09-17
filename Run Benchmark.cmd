@echo off
setlocal
if not exist "%~dp0benchmarks\.venv\Scripts\python.exe" (
  echo The benchmark environment was not found.
  echo See benchmarks\START_HERE.md for setup instructions.
  pause
  exit /b 1
)
"%~dp0benchmarks\.venv\Scripts\python.exe" "%~dp0benchmarks\baseline.py" %*
set "benchmark_exit=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %benchmark_exit%

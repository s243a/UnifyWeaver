@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM UnifyWeaver Test Environment Launcher
REM Prefers Windows Terminal > ConEmu > Standard Console
REM Updated to use new terminal launcher
REM ============================================================================

set "TEST_ROOT=%~dp0"
if "%TEST_ROOT:~-1%"=="\" set "TEST_ROOT=%TEST_ROOT:~0,-1%"

echo [UnifyWeaver] Test Environment
echo [UnifyWeaver] Location: %TEST_ROOT%
echo [INFO] Using new terminal launcher (prefers Windows Terminal)
echo.

REM --- Launch with new PowerShell launcher ---
powershell.exe -ExecutionPolicy Bypass -File "%TEST_ROOT%\Start-UnifyWeaverTerminal.ps1"

endlocal

@echo off
REM ============================================================================
REM UnifyWeaver Test Environment Launcher
REM Prefers Windows Terminal > ConEmu > PowerShell
REM ============================================================================

powershell.exe -ExecutionPolicy Bypass -File "%~dp0Start-UnifyWeaverTerminal.ps1"

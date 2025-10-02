@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM start_unifyweaver_windows.bat - Native Windows launcher for UnifyWeaver
REM
REM Pure Windows solution - uses native Windows SWI-Prolog (no Cygwin/WSL)
REM Supports ConEmu terminal emulator for enhanced experience
REM ============================================================================

REM --- Resolve project root ---
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Normalize trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo [UnifyWeaver] Starting native Windows environment...
echo [UnifyWeaver] Project root: %PROJECT_ROOT%

REM --- Check for SWI-Prolog ---
where swipl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] swipl not found on PATH.
  echo [ERROR] Please install SWI-Prolog for Windows from https://www.swi-prolog.org/download/stable
  echo [ERROR] Make sure to add it to your PATH during installation.
  pause
  exit /b 1
)

REM Get SWI-Prolog version for verification
for /f "tokens=*" %%V in ('swipl --version 2^>^&1 ^| findstr /C:"SWI-Prolog"') do (
  echo [UnifyWeaver] Found: %%V
)

REM --- Optional: Check for ConEmu ---
set "CONEMU_EXE="

REM Check common ConEmu locations
if exist "%ProgramFiles%\ConEmu\ConEmu64.exe" (
  set "CONEMU_EXE=%ProgramFiles%\ConEmu\ConEmu64.exe"
)
if exist "%ProgramFiles(x86)%\ConEmu\ConEmu.exe" (
  set "CONEMU_EXE=%ProgramFiles(x86)%\ConEmu\ConEmu.exe"
)
if exist "%LocalAppData%\ConEmu\ConEmu64.exe" (
  set "CONEMU_EXE=%LocalAppData%\ConEmu\ConEmu64.exe"
)

REM --- Convert Windows paths for Prolog (use forward slashes) ---
set "WIN_PROJECT_ROOT=%PROJECT_ROOT%"
set "PROLOG_ROOT=%PROJECT_ROOT:\=/%"
set "PROLOG_SRC=%PROLOG_ROOT%/src"
set "PROLOG_UNIFYWEAVER=%PROLOG_ROOT%/src/unifyweaver"

REM --- Build Prolog initialization goal ---
REM Note: Use \n for newlines in Prolog strings
set PROLOG_GOAL=( working_directory(CWD, CWD), ^
  format('[UnifyWeaver] Working directory: ~~w~~n', [CWD]), ^
  atom_string('%PROLOG_ROOT%', RootStr), ^
  atom_concat(RootStr, '/src', AbsSrcDir), ^
  atom_concat(RootStr, '/src/unifyweaver', AbsUnifyweaverDir), ^
  asserta(user:library_directory(AbsSrcDir)), ^
  asserta(file_search_path(unifyweaver, AbsUnifyweaverDir)), ^
  format('[UnifyWeaver] Library paths configured~~n', []), ^
  format('[UnifyWeaver] Native Windows environment ready!~~n~~n', []), ^
  format('Helper commands:~~n', []), ^
  format('  load_stream      - Load stream compiler~~n', []), ^
  format('  load_recursive   - Load recursive compiler~~n', []), ^
  format('  test_stream      - Test stream compiler~~n', []), ^
  format('  test_recursive   - Test recursive compiler~~n', []), ^
  format('  test_advanced    - Test advanced recursion~~n~~n', []), ^
  asserta((load_stream :- ^
    (use_module(unifyweaver(core/stream_compiler)) -^> ^
      format('stream_compiler loaded successfully!~~n', []) ^
    ; format('Failed to load stream_compiler~~n', [])))), ^
  asserta((load_recursive :- ^
    (use_module(unifyweaver(core/recursive_compiler)) -^> ^
      format('recursive_compiler loaded successfully!~~n', []) ^
    ; format('Failed to load recursive_compiler~~n', [])))), ^
  asserta((test_stream :- ^
    use_module(unifyweaver(core/stream_compiler)), ^
    test_stream_compiler)), ^
  asserta((test_recursive :- ^
    use_module(unifyweaver(core/recursive_compiler)), ^
    test_recursive_compiler)), ^
  asserta((test_advanced :- ^
    use_module(unifyweaver(core/advanced/test_advanced)), ^
    test_all_advanced)) ^
)

REM --- Change to project root ---
pushd "%PROJECT_ROOT%"

REM --- Launch with ConEmu if available, else standard console ---
if defined CONEMU_EXE (
  echo [UnifyWeaver] Launching in ConEmu terminal...
  "%CONEMU_EXE%" -Dir "%PROJECT_ROOT%" -run cmd /k swipl -q -g "%PROLOG_GOAL%" -t prolog
) else (
  echo [UnifyWeaver] ConEmu not found - using standard console
  echo [UnifyWeaver] For better terminal experience, install ConEmu from https://conemu.github.io/
  echo.
  swipl -q -g "%PROLOG_GOAL%" -t prolog
)

popd
endlocal

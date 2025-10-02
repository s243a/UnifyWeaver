@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM UnifyWeaver Windows Launcher (Cygwin/MSYS2/WSL)
REM Resolve Windows path to the scripts directory (this .bat lives here)
set "WIN_SCRIPTS=%~dp0"
if "%WIN_SCRIPTS:~-1%"=="\" set "WIN_SCRIPTS=%WIN_SCRIPTS:~0,-1%"

REM Add SWI-Prolog to PATH if it exists
if exist "C:\Program Files\swipl\bin" (
    set "PATH=C:\Program Files\swipl\bin;%PATH%"
)
set SWIPLT=utf8

REM Prefer Cygwin, then MSYS2, else WSL
set "CYG_BASH=C:\cygwin64\bin\bash.exe"
set "CYG_CYGPATH=C:\cygwin64\bin\cygpath.exe"
set "MSYS_BASH=C:\msys64\usr\bin\bash.exe"
set "MSYS_CYGPATH=C:\msys64\usr\bin\cygpath.exe"

REM --- Cygwin ---
if exist "%CYG_BASH%" (
  if not exist "%CYG_CYGPATH%" (
    echo [ERROR] cygpath.exe not found under Cygwin. >&2
    exit /b 1
  )
  echo [UnifyWeaver] Using Cygwin bash
  for /f "usebackq delims=" %%S in (`%CYG_CYGPATH% -u "%WIN_SCRIPTS%" 2^>^&1`) do (
    set "POSIX_SCRIPTS=%%S"
  )
  echo [UnifyWeaver] Scripts directory: !POSIX_SCRIPTS!
  "%CYG_BASH%" -lc "cd \"!POSIX_SCRIPTS!\"; bash ./unifyweaver_console.sh"
  exit /b %ERRORLEVEL%
)

REM --- MSYS2 ---
if exist "%MSYS_BASH%" (
  if not exist "%MSYS_CYGPATH%" (
    echo [ERROR] cygpath.exe not found under MSYS2. >&2
    exit /b 1
  )
  echo [UnifyWeaver] Using MSYS2 bash
  for /f "usebackq delims=" %%S in (`"%MSYS_CYGPATH%" -u "%WIN_SCRIPTS%"`) do set "POSIX_SCRIPTS=%%S"
  "%MSYS_BASH%" -lc "cd \"%POSIX_SCRIPTS%\"; bash ./unifyweaver_console.sh"
  exit /b %ERRORLEVEL%
)

REM --- WSL fallback ---
where wsl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Could not find Cygwin, MSYS2, or WSL. >&2
  echo [ERROR] Please install one of these to use UnifyWeaver. >&2
  pause
  exit /b 1
)
echo [UnifyWeaver] Using WSL
for /f "usebackq delims=" %%S in (`wsl wslpath -a "%WIN_SCRIPTS%"`) do set "WSL_SCRIPTS=%%S"
wsl bash -lc "cd \"%WSL_SCRIPTS%\"; bash ./unifyweaver_console.sh"
exit /b %ERRORLEVEL%

#try {
    Write-Host "==================================="
    Write-Host "--- Test Environment Setup ---"
    # We are already in the correct directory, so no Set-Location is needed.

    Write-Host "==================================="
    Write-Host "--- Sourcing Compatibility Layer ---"
    . "$PSScriptRoot\scripts\init_unify_compat.ps1"

    Write-Host "==================================="
    Write-Host "--- TEST 1: WSL Backend Verification ---"
    $env:UNIFYWEAVER_EXEC_MODE = 'wsl'
    $output_wsl = uw-uname -o
    if ($output_wsl -like '*Linux*') { Write-Host 'PASS: WSL backend confirmed.' -ForegroundColor Green }
    else { Write-Host "FAIL: WSL backend not detected. Output: $output_wsl" -ForegroundColor Red }

    Write-Host "==================================="
    Write-Host "--- TEST 2: Cygwin Backend Verification ---"
    $env:UNIFYWEAVER_EXEC_MODE = 'cygwin'
    $output_cygwin = uw-uname -o
    if ($output_cygwin -like '*Cygwin*') { Write-Host 'PASS: Cygwin backend confirmed.' -ForegroundColor Green }
    else { Write-Host "FAIL: Cygwin backend not detected. Output: $output_cygwin" -ForegroundColor Red }

    Write-Host "==================================="
    Write-Host "--- TEST 3: Pipeline Verification (ls | grep) ---"
    $env:UNIFYWEAVER_EXEC_MODE = 'cygwin' # Use cygwin backend for this test
    $output_pipe = uw-ls | uw-grep 'examples'
    if ($output_pipe -like '*examples*') { Write-Host 'PASS: Pipeline and wrappers (ls | grep) are working.' -ForegroundColor Green }
    else { Write-Host "FAIL: Pipeline test did not produce expected output. Output: $output_pipe" -ForegroundColor Red }

    Write-Host "==================================="
    Write-Host "--- ALL TESTS COMPLETED SUCCESSFULLY ---" -ForegroundColor Green
#}
#catch {
#    Write-Host "--- SCRIPT FAILED ---" -ForegroundColor Red
#    # Write the actual error record to the host
#    Write-Host $_.Exception.ToString() -ForegroundColor Red
#    Write-Host $_.ScriptStackTrace -ForegroundColor Red
#}

# Fix-WindowsTerminalEmoji.ps1
# Diagnose and fix Windows Terminal font for emoji support

param(
    [switch]$AutoFix
)

Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Windows Terminal Emoji Font Diagnostic              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Test 1: Check if emoji render correctly
Write-Host "[Test 1] Emoji Rendering Test" -ForegroundColor Yellow
Write-Host "BMP Symbols (should work): ✅ ❌ ⚠ ℹ ⚡" -ForegroundColor White
Write-Host "Non-BMP Emoji (need font fix): 🚀 📊 📈 💾 🎉" -ForegroundColor White
Write-Host ""
Write-Host "Question: Do you see actual emoji above (not boxes or Asian characters)?" -ForegroundColor Yellow
Write-Host "If you see garbled text or boxes for the second line, continue with this script." -ForegroundColor Yellow
Write-Host ""

# Test 2: Find Windows Terminal settings
Write-Host "[Test 2] Locating Windows Terminal Settings" -ForegroundColor Yellow
$settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

if (Test-Path $settingsPath) {
    Write-Host "[✓] Found settings at: $settingsPath" -ForegroundColor Green
} else {
    Write-Host "[✗] Settings not found at expected path" -ForegroundColor Red
    Write-Host "[INFO] Searching for alternative locations..." -ForegroundColor Yellow

    $altPath = Get-ChildItem "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal*\LocalState\settings.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($altPath) {
        $settingsPath = $altPath.FullName
        Write-Host "[✓] Found settings at: $settingsPath" -ForegroundColor Green
    } else {
        Write-Host "[✗] Could not find Windows Terminal settings file" -ForegroundColor Red
        Write-Host "[INFO] Make sure Windows Terminal is installed" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host ""

# Test 3: Read current font settings
Write-Host "[Test 3] Current Font Configuration" -ForegroundColor Yellow
try {
    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json

    # Find PowerShell profile (try multiple profile names)
    $psProfile = $settings.profiles.list | Where-Object {
        $_.name -in @("Windows PowerShell", "PowerShell", "pwsh")
    } | Select-Object -First 1

    if ($psProfile) {
        Write-Host "[✓] Found PowerShell profile: $($psProfile.name)" -ForegroundColor Green

        if ($psProfile.font -and $psProfile.font.face) {
            $currentFont = $psProfile.font.face
            Write-Host "[INFO] Current font: $currentFont" -ForegroundColor Cyan

            # Check if it's a good emoji font
            $goodFonts = @("Cascadia Code", "Cascadia Mono", "Consolas", "Courier New")
            if ($currentFont -in $goodFonts) {
                Write-Host "[✓] Font supports emoji via fallback" -ForegroundColor Green
                Write-Host "[INFO] If emoji still don't render, try restarting Windows Terminal" -ForegroundColor Yellow
            } else {
                Write-Host "[⚠] Font '$currentFont' may not support emoji properly" -ForegroundColor Yellow
                Write-Host "[INFO] Recommended: Change to 'Cascadia Code'" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[INFO] No custom font set (using default)" -ForegroundColor Cyan
            Write-Host "[INFO] Default should work, but explicitly setting 'Cascadia Code' is recommended" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[✗] Could not find PowerShell profile in settings" -ForegroundColor Red
    }
} catch {
    Write-Host "[✗] Error reading settings: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: Check encoding
Write-Host "[Test 4] Console Encoding" -ForegroundColor Yellow
Write-Host "[INFO] Output Encoding: $([Console]::OutputEncoding.EncodingName)" -ForegroundColor Cyan
Write-Host "[INFO] Input Encoding: $([Console]::InputEncoding.EncodingName)" -ForegroundColor Cyan
if ([Console]::OutputEncoding.CodePage -eq 65001) {
    Write-Host "[✓] UTF-8 encoding is active" -ForegroundColor Green
} else {
    Write-Host "[⚠] Not using UTF-8 (code page: $([Console]::OutputEncoding.CodePage))" -ForegroundColor Yellow
}
Write-Host ""

# Auto-fix option
if ($AutoFix) {
    Write-Host "[Auto-Fix] Updating Windows Terminal Font" -ForegroundColor Yellow
    try {
        # Backup original settings
        $backupPath = "$settingsPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $settingsPath $backupPath
        Write-Host "[✓] Created backup: $backupPath" -ForegroundColor Green

        # Update font
        if ($psProfile) {
            if (-not $psProfile.font) {
                $psProfile | Add-Member -NotePropertyName "font" -NotePropertyValue @{ face = "Cascadia Code" } -Force
            } else {
                $psProfile.font.face = "Cascadia Code"
            }

            # Save settings
            $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath
            Write-Host "[✓] Font updated to 'Cascadia Code'" -ForegroundColor Green
            Write-Host "[INFO] Please restart Windows Terminal for changes to take effect" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[✗] Auto-fix failed: $_" -ForegroundColor Red
        Write-Host "[INFO] Please change font manually (Ctrl+, → Appearance → Font face → Cascadia Code)" -ForegroundColor Yellow
    }
} else {
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "Manual Fix Instructions:" -ForegroundColor Yellow
    Write-Host "1. Press Ctrl+, in Windows Terminal" -ForegroundColor White
    Write-Host "2. Click your PowerShell profile" -ForegroundColor White
    Write-Host "3. Under 'Appearance' → 'Font face'" -ForegroundColor White
    Write-Host "4. Select 'Cascadia Code'" -ForegroundColor White
    Write-Host "5. Click 'Save' and restart Windows Terminal" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run this script with -AutoFix to update automatically:" -ForegroundColor Yellow
    Write-Host "  .\Fix-WindowsTerminalEmoji.ps1 -AutoFix" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[Done] Diagnostic complete" -ForegroundColor Green
Write-Host ""

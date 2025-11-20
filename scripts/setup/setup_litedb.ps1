# scripts/setup/setup_litedb.ps1
# Interactive LiteDB installation for local and global environments

param()

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "../..")
Set-Location $projectRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "LiteDB Setup for UnifyWeaver" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will help you install LiteDB:"
Write-Host "  • Local: Project lib/ directory"
Write-Host "  • Global: System NuGet cache (for all .NET projects)"
Write-Host ""

# Detect .NET SDK
try {
    $latestSdk = dotnet --list-sdks | ForEach-Object { $_.Split(' ')[0] } |
                 Sort-Object -Descending | Select-Object -First 1
    $sdkMajor = [int]$latestSdk.Split('.')[0]
    Write-Host "Detected .NET SDK: $latestSdk" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: .NET SDK not found. Please install .NET SDK first." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Determine target framework
$target = switch ($sdkMajor) {
    { $_ -ge 8 } { "net8.0"; break }
    { $_ -ge 6 } { "net6.0"; break }
    default { "netstandard2.0" }
}

# ==========================================
# Local Installation Menu
# ==========================================
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Local Installation (lib/)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Choose version to install locally:"
Write-Host "  1) None (skip local installation)"
Write-Host "  2) LiteDB 5.0.21 (stable)"
Write-Host "  3) LiteDB 6.0.0-prerelease.73 (experimental)"
Write-Host ""
$localChoice = Read-Host "Select option [1-3]"
Write-Host ""

# ==========================================
# Global Installation Menu
# ==========================================
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Global Installation (NuGet cache)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Choose version to install globally:"
Write-Host "  1) None (skip global installation)"
Write-Host "  2) LiteDB 5.0.21 (stable)"
Write-Host "  3) LiteDB 6.0.0-prerelease.73 (experimental)"
Write-Host ""
$globalChoice = Read-Host "Select option [1-3]"
Write-Host ""

# ==========================================
# Confirmation
# ==========================================
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Confirmation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "You selected:"

switch ($localChoice) {
    "1" { Write-Host "  Local:  None" }
    "2" { Write-Host "  Local:  LiteDB 5.0.21 (stable)" }
    "3" { Write-Host "  Local:  LiteDB 6.0.0-prerelease.73 (prerelease)" }
    default { Write-Host "  Local:  Invalid choice" -ForegroundColor Red; exit 1 }
}

switch ($globalChoice) {
    "1" { Write-Host "  Global: None" }
    "2" { Write-Host "  Global: LiteDB 5.0.21 (stable)" }
    "3" { Write-Host "  Global: LiteDB 6.0.0-prerelease.73 (prerelease)" }
    default { Write-Host "  Global: Invalid choice" -ForegroundColor Red; exit 1 }
}

Write-Host ""
$confirm = Read-Host "Proceed with installation? [y/N]"

if ($confirm -notmatch "^[Yy]$") {
    Write-Host "Installation cancelled."
    exit 0
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installing..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ==========================================
# Helper Functions
# ==========================================

function Install-LocalLiteDB {
    param(
        [string]$Version,
        [bool]$IsBeta
    )

    Write-Host "[Local] Installing LiteDB $Version..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path "lib/litedb-temp" | Out-Null
    Set-Location "lib/litedb-temp"

    Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/LiteDB/$Version" `
        -OutFile "litedb.nupkg"

    Rename-Item "litedb.nupkg" "litedb.zip"
    Expand-Archive -Path "litedb.zip" -DestinationPath "." -Force

    # Try to find DLL in order: detected target, netstandard2.0, netstandard1.3
    $dllTarget = $null
    $tryTargets = @($target, "netstandard2.0", "netstandard1.3")
    foreach ($tryTarget in $tryTargets) {
        if (Test-Path "lib/$tryTarget/LiteDB.dll") {
            $dllTarget = $tryTarget
            break
        }
    }

    if ($dllTarget) {
        if ($IsBeta) {
            Copy-Item "lib/$dllTarget/LiteDB.dll" "../LiteDB-beta.dll"
            Write-Host "✅ LiteDB $Version installed to: lib/LiteDB-beta.dll (from $dllTarget)" -ForegroundColor Green
        } else {
            Copy-Item "lib/$dllTarget/LiteDB.dll" "../LiteDB.dll"
            Write-Host "✅ LiteDB $Version installed to: lib/LiteDB.dll (from $dllTarget)" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ Error: Could not find LiteDB.dll in package" -ForegroundColor Red
        Write-Host "Available targets:"
        Get-ChildItem "lib/"
        Set-Location "../.."
        Remove-Item "lib/litedb-temp" -Recurse -Force
        exit 1
    }

    Set-Location "../.."
    Remove-Item "lib/litedb-temp" -Recurse -Force
}

function Install-GlobalLiteDB {
    param([string]$Version)

    Write-Host "[Global] Installing LiteDB $Version to NuGet cache..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path "tmp/litedb-global" | Out-Null
    Set-Location "tmp/litedb-global"

    @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$target</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="LiteDB" Version="$Version" />
  </ItemGroup>
</Project>
"@ | Out-File -FilePath "litedb-global.csproj"

    "class Program { static void Main() {} }" | Out-File -FilePath "Program.cs"

    dotnet restore | Out-Null
    dotnet build | Out-Null

    $nugetCache = "$env:USERPROFILE\.nuget\packages\litedb\$($Version.ToLower())"
    if (Test-Path $nugetCache) {
        Write-Host "✅ LiteDB $Version installed to NuGet cache:" -ForegroundColor Green
        Write-Host "   $nugetCache" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Warning: Could not verify NuGet cache installation" -ForegroundColor Yellow
    }

    Set-Location "../.."
    Remove-Item "tmp/litedb-global" -Recurse -Force
}

# ==========================================
# Execute Installations
# ==========================================

# Local installation
switch ($localChoice) {
    "1" { Write-Host "[Local] Skipping local installation" }
    "2" { Install-LocalLiteDB -Version "5.0.21" -IsBeta $false }
    "3" { Install-LocalLiteDB -Version "6.0.0-prerelease.73" -IsBeta $true }
}

Write-Host ""

# Global installation
switch ($globalChoice) {
    "1" { Write-Host "[Global] Skipping global installation" }
    "2" { Install-GlobalLiteDB -Version "5.0.21" }
    "3" { Install-GlobalLiteDB -Version "6.0.0-prerelease.73" }
}

# ==========================================
# Summary
# ==========================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "lib/LiteDB.dll") {
    Write-Host "📦 lib/LiteDB.dll (stable)" -ForegroundColor White
}

if (Test-Path "lib/LiteDB-beta.dll") {
    Write-Host "📦 lib/LiteDB-beta.dll (beta)" -ForegroundColor White
}

Write-Host ""
Write-Host "Usage in Prolog code:" -ForegroundColor Yellow
Write-Host "  dll_references(['lib/LiteDB.dll'])        # stable"
Write-Host "  dll_references(['lib/LiteDB-beta.dll'])   # beta"
Write-Host ""
Write-Host "To switch versions later, run:" -ForegroundColor Yellow
Write-Host "  .\scripts\setup\switch_litedb_version.ps1"
Write-Host ""

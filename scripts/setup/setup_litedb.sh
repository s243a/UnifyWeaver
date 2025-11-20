#!/bin/bash
# scripts/setup/setup_litedb.sh
# Interactive LiteDB installation for local and global environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "LiteDB Setup for UnifyWeaver"
echo "=========================================="
echo ""
echo "This script will help you install LiteDB:"
echo "  • Local: Project lib/ directory"
echo "  • Global: System NuGet cache (for all .NET projects)"
echo ""

# Detect .NET SDK
LATEST_SDK=$(dotnet --list-sdks 2>/dev/null | awk '{print $1}' | sort -V | tail -n 1)
if [ -z "$LATEST_SDK" ]; then
    echo "❌ Error: .NET SDK not found. Please install .NET SDK first."
    exit 1
fi
SDK_MAJOR=${LATEST_SDK%%.*}
echo "Detected .NET SDK: $LATEST_SDK"
echo ""

# Determine target framework
if [ "$SDK_MAJOR" -ge 8 ]; then
    TARGET="net8.0"
elif [ "$SDK_MAJOR" -ge 6 ]; then
    TARGET="net6.0"
else
    TARGET="netstandard2.0"
fi

# ==========================================
# Local Installation Menu
# ==========================================
echo "=========================================="
echo "Local Installation (lib/)"
echo "=========================================="
echo "Choose version to install locally:"
echo "  1) None (skip local installation)"
echo "  2) LiteDB 5.0.21 (stable)"
echo "  3) LiteDB 6.0.0-prerelease.73 (experimental)"
echo ""
read -p "Select option [1-3]: " LOCAL_CHOICE
echo ""

# ==========================================
# Global Installation Menu
# ==========================================
echo "=========================================="
echo "Global Installation (NuGet cache)"
echo "=========================================="
echo "Choose version to install globally:"
echo "  1) None (skip global installation)"
echo "  2) LiteDB 5.0.21 (stable)"
echo "  3) LiteDB 6.0.0-prerelease.73 (experimental)"
echo ""
read -p "Select option [1-3]: " GLOBAL_CHOICE
echo ""

# ==========================================
# Confirmation
# ==========================================
echo "=========================================="
echo "Confirmation"
echo "=========================================="
echo "You selected:"

case $LOCAL_CHOICE in
    1) echo "  Local:  None" ;;
    2) echo "  Local:  LiteDB 5.0.21 (stable)" ;;
    3) echo "  Local:  LiteDB 6.0.0-prerelease.73 (prerelease)" ;;
    *) echo "  Local:  Invalid choice"; exit 1 ;;
esac

case $GLOBAL_CHOICE in
    1) echo "  Global: None" ;;
    2) echo "  Global: LiteDB 5.0.21 (stable)" ;;
    3) echo "  Global: LiteDB 6.0.0-prerelease.73 (prerelease)" ;;
    *) echo "  Global: Invalid choice"; exit 1 ;;
esac

echo ""
read -p "Proceed with installation? [y/N]: " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "Installing..."
echo "=========================================="
echo ""

# ==========================================
# Helper Functions
# ==========================================

install_local() {
    local VERSION=$1
    local IS_BETA=$2

    echo "[Local] Installing LiteDB $VERSION..."

    mkdir -p lib/litedb-temp
    cd lib/litedb-temp

    curl -L -o litedb.nupkg \
        "https://www.nuget.org/api/v2/package/LiteDB/$VERSION"

    unzip -q litedb.nupkg

    # Determine which target to use based on what's available
    # Try in order: net8.0, net6.0, netstandard2.0, netstandard1.3
    DLL_TARGET=""
    for try_target in "$TARGET" "netstandard2.0" "netstandard1.3"; do
        if [ -f "lib/$try_target/LiteDB.dll" ]; then
            DLL_TARGET="$try_target"
            break
        fi
    done

    if [ -n "$DLL_TARGET" ]; then
        if [ "$IS_BETA" == "true" ]; then
            cp "lib/$DLL_TARGET/LiteDB.dll" ../LiteDB-beta.dll
            echo "✅ LiteDB $VERSION installed to: lib/LiteDB-beta.dll (from $DLL_TARGET)"
        else
            cp "lib/$DLL_TARGET/LiteDB.dll" ../LiteDB.dll
            echo "✅ LiteDB $VERSION installed to: lib/LiteDB.dll (from $DLL_TARGET)"
        fi
    else
        echo "❌ Error: Could not find LiteDB.dll in package"
        echo "Available targets:"
        ls -la lib/
        cd ../..
        rm -rf lib/litedb-temp
        exit 1
    fi

    cd ../..
    rm -rf lib/litedb-temp
}

install_global() {
    local VERSION=$1

    echo "[Global] Installing LiteDB $VERSION to NuGet cache..."

    mkdir -p tmp/litedb-global
    cd tmp/litedb-global

    cat > litedb-global.csproj <<EOF
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TARGET</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="LiteDB" Version="$VERSION" />
  </ItemGroup>
</Project>
EOF

    echo "class Program { static void Main() {} }" > Program.cs

    dotnet restore > /dev/null 2>&1
    dotnet build > /dev/null 2>&1

    NUGET_CACHE="$HOME/.nuget/packages/litedb/${VERSION,,}"
    if [ -d "$NUGET_CACHE" ]; then
        echo "✅ LiteDB $VERSION installed to NuGet cache:"
        echo "   $NUGET_CACHE"
    else
        echo "⚠️  Warning: Could not verify NuGet cache installation"
    fi

    cd ../..
    rm -rf tmp/litedb-global
}

# ==========================================
# Execute Installations
# ==========================================

# Local installation
case $LOCAL_CHOICE in
    1) echo "[Local] Skipping local installation" ;;
    2) install_local "5.0.21" "false" ;;
    3) install_local "6.0.0-prerelease.73" "true" ;;
esac

echo ""

# Global installation
case $GLOBAL_CHOICE in
    1) echo "[Global] Skipping global installation" ;;
    2) install_global "5.0.21" ;;
    3) install_global "6.0.0-prerelease.73" ;;
esac

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""

if [ -f "lib/LiteDB.dll" ]; then
    echo "📦 lib/LiteDB.dll (stable)"
fi

if [ -f "lib/LiteDB-beta.dll" ]; then
    echo "📦 lib/LiteDB-beta.dll (beta)"
fi

echo ""
echo "Usage in Prolog code:"
echo "  dll_references(['lib/LiteDB.dll'])        % stable"
echo "  dll_references(['lib/LiteDB-beta.dll'])   % beta"
echo ""
echo "To switch versions later, run:"
echo "  bash scripts/setup/switch_litedb_version.sh"
echo ""

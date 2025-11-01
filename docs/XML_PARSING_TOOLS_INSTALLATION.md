# XML Parsing Tools Installation Guide

This guide covers installation of XML parsing tools for the `xml_source` plugin on various platforms.

## Overview

The `xml_source` plugin supports multiple parsing engines:

1. **lxml (Python)** - Recommended for large files (true streaming, constant memory)
2. **xmllint** - Standard Linux tool (part of libxml2-utils)
3. **xmlstarlet** - Alternative CLI tool (NOT recommended for large files due to memory issues)

## Installation by Platform

### Linux (Debian/Ubuntu)

#### Install lxml (Recommended)
```bash
# Using pip (Python package manager)
pip3 install lxml

# Or using system package manager
sudo apt-get update
sudo apt-get install python3-lxml
```

#### Install xmllint
```bash
sudo apt-get update
sudo apt-get install libxml2-utils
```

#### Install xmlstarlet (Optional)
```bash
sudo apt-get update
sudo apt-get install xmlstarlet
```

### Linux (RHEL/CentOS/Fedora)

#### Install lxml
```bash
# Using pip
pip3 install lxml

# Or using system package manager
sudo dnf install python3-lxml    # Fedora
sudo yum install python3-lxml    # RHEL/CentOS
```

#### Install xmllint
```bash
sudo dnf install libxml2         # Fedora
sudo yum install libxml2         # RHEL/CentOS
```

#### Install xmlstarlet
```bash
sudo dnf install xmlstarlet      # Fedora
sudo yum install xmlstarlet      # RHEL/CentOS
```

### Linux (Arch)

#### Install lxml
```bash
# Using pip
pip3 install lxml

# Or using system package manager
sudo pacman -S python-lxml
```

#### Install xmllint
```bash
sudo pacman -S libxml2
```

#### Install xmlstarlet
```bash
sudo pacman -S xmlstarlet
```

### macOS

#### Install lxml
```bash
# Using pip
pip3 install lxml

# Or using Homebrew
brew install libxml2 libxslt
pip3 install lxml
```

#### Install xmllint
```bash
# xmllint comes pre-installed on macOS
# To update to latest version:
brew install libxml2
```

#### Install xmlstarlet
```bash
brew install xmlstarlet
```

### Windows (WSL - Windows Subsystem for Linux)

Use the Linux (Ubuntu) instructions above within your WSL environment.

#### WSL PATH priority fix

When running the UnifyWeaver test suite from PowerShell (or another Windows shell) the inherited `PATH` may list Windows directories before the WSL ones, causing tools such as `xmlstarlet` to resolve to Windows shims or not be found. To align the environment with what WSL expects:

```bash
source scripts/testing/reorder-path.sh
```

This script reorders `$PATH` so the Linux directories from `/etc/environment` take precedence. Every automated test run that shells out from the Windows side should source it before invoking `swipl`.

### Windows (Native - via Cygwin)

#### Install lxml

⚠️ **Important:** On Cygwin, `pip3 install lxml` will FAIL unless you install development packages first. The easiest approach is to use the pre-compiled binary.

**Option 1: Pre-compiled Binary (EASIEST - Recommended)**
1. Close Cygwin terminal
2. Run Cygwin's `setup-x86_64.exe` or `setup-x86.exe`
3. Search for and select:
   - `python39-lxml` (or `python3-lxml` - pre-compiled lxml package)

   **Note:** The package name may vary by Cygwin version. Search for "lxml" in the package list.

4. Complete installation
5. Open Cygwin terminal and verify:
   ```bash
   python3 -c "import lxml; print('lxml version:', lxml.__version__)"
   ```

**Option 2: Compile from Source (Advanced - Requires Multiple Packages)**

⚠️ **Only use this if pre-compiled package is unavailable**

1. Close Cygwin terminal
2. Run Cygwin's `setup-x86_64.exe` or `setup-x86.exe`
3. Search for and install ALL of these packages:
   - `python39` (or latest Python 3.x)
   - `python39-devel` (Python development headers - REQUIRED)
   - `libxml2-devel` (XML library headers - REQUIRED)
   - `libxslt-devel` (XSLT library headers - REQUIRED)
   - `gcc-core` (C compiler - REQUIRED)
   - `gcc-g++` (C++ compiler - may be needed)
   - `zlib-devel` (compression library - REQUIRED)
   - `make` (build tool)

4. Complete installation and restart Cygwin terminal
5. Verify packages are installed:
   ```bash
   # Check for development headers
   ls /usr/include/libxml2/libxml/
   ls /usr/include/libxslt/
   ```

6. Now try pip:
   ```bash
   pip3 install lxml
   ```

**If compilation still fails:** Use Option 1 (pre-compiled binary) instead

#### Install xmllint
```bash
# Install via Cygwin setup.exe
# Package: libxml2
```

#### Install xmlstarlet
```bash
# Install via Cygwin setup.exe
# Package: xmlstarlet
```

### Windows (Native - via Chocolatey)

#### Install Python and lxml
```powershell
# From PowerShell (as Administrator)
choco install python3
pip3 install lxml
```

#### Install xmlstarlet
```powershell
choco install xmlstarlet
```

## Verification

### Verify lxml installation
```bash
python3 -c "import lxml; print('lxml version:', lxml.__version__)"
```

Expected output:
```
lxml version: 5.x.x
```

### Verify xmllint installation
```bash
xmllint --version
```

Expected output:
```
xmllint: using libxml version 20xxx
   compiled with: Threads Tree Output Push Reader Patterns Writer SAXv1 FTP HTTP DTDValid HTML Legacy C14N Catalog XPath XPointer XInclude Iconv ICU ISO8859X Unicode Regexps Automata Schemas Schematron Modules Debug Zlib Lzma
```

**Understanding version numbers:**
- Version shown is in hex format
- Example: `20910` = libxml2 version 2.9.10 (2.09.10 in hex)
- Example: `21004` = libxml2 version 2.10.4

**Check package version (Linux only):**
```bash
# Debian/Ubuntu (WSL)
dpkg -l | grep libxml2-utils

# RHEL/CentOS/Fedora
rpm -q libxml2

# Arch
pacman -Q libxml2
```

**Check package version (Cygwin):**
```bash
# Using cygcheck
cygcheck -c libxml2

# Or search Cygwin packages
cygcheck -p xmllint
```

### Verify xmlstarlet installation
```bash
xmlstarlet --version
```

Expected output:
```
1.6.1
compiled against libxml2 2.x.x, linked with 20xxx
compiled against libxslt 1.x.x, linked with 10xxx
```

### Verify xmllint extraction (project)
```bash
source scripts/testing/reorder-path.sh
swipl -s tests/core/test_xml_source.pl \
      -g "run_tests([xml_source:xmllint_extraction])" \
      -t halt
```

The test prints each extracted record to stdout so you can confirm namespaces and CDATA blocks look correct.

To compare the output when namespace repair is disabled, run:

```bash
swipl -s tests/core/test_xml_source.pl \
      -g "run_tests([xml_source:xmllint_extraction_without_namespace_fix])" \
      -t halt
```

This variant demonstrates the behaviour with `namespace_fix(false)` set in the source configuration.

## Troubleshooting

### Quick Reference - Common Errors

| Error Message | Platform | Solution |
|---------------|----------|----------|
| `Error: Please make sure the libxml2 and libxslt development packages are installed` | WSL/Ubuntu | [Install dev packages](#lxml-installation-fails-on-wsl-windows-subsystem-for-linux) |
| `Error: Please make sure the libxml2 and libxslt development packages are installed` | Cygwin | [Use Cygwin Setup](#lxml-installation-fails-on-cygwin) |
| `Getting requirements to build wheel did not run successfully` | WSL/Linux | [Install dev packages](#lxml-installation-fails-on-wsl-windows-subsystem-for-linux) |
| `pip3: command not found` | WSL | [Install pip](#wsl-specific-issues) |
| `Permission denied` | Any | [Use --user flag](#permission-denied-errors) |
| Compilation hangs/takes forever | Cygwin | [Use pre-compiled binary](#lxml-installation-fails-on-cygwin) |

### lxml installation fails on WSL (Windows Subsystem for Linux)

**Error:** `Error: Please make sure the libxml2 and libxslt development packages are installed.`

This is the most common issue when installing lxml on WSL/Ubuntu. The error occurs because lxml needs to compile C extensions and requires development libraries.

**Solution:**
```bash
# Install all required development packages
sudo apt-get update
sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev zlib1g-dev

# Now install lxml
pip3 install lxml
```

**Alternative (pre-compiled binary):**
```bash
# Use the system package instead of pip
sudo apt-get install python3-lxml
```

**Verify installation:**
```bash
python3 -c "import lxml; print('lxml version:', lxml.__version__)"
```

### lxml installation fails on Cygwin

⚠️ **This is a very common issue on Cygwin. The pre-compiled binary approach is strongly recommended.**

**Error:** `Error: Please make sure the libxml2 and libxslt development packages are installed.`

```
Collecting lxml
  Using cached lxml-6.0.2.tar.gz (4.1 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [3 lines of output]
      Building lxml version 6.0.2.
      Building without Cython.
      Error: Please make sure the libxml2 and libxslt development packages are installed.
```

**This error means:** Cygwin's pip is trying to compile lxml from source, but the C development libraries are missing.

**Solution 1: Use Pre-compiled Binary (EASIEST - Recommended)**

This completely avoids compilation and pip:

1. **Close Cygwin terminal**
2. **Run Cygwin Setup:** `setup-x86_64.exe` (usually in `C:\cygwin64\`)
3. **Search for lxml:**
   - In the package search box, type: `lxml`
   - Look for: `python39-lxml` or `python3-lxml`
   - Click the "Skip" button to change it to a version number (this installs it)
4. **Complete installation** (click Next through the dialogs)
5. **Open Cygwin terminal and verify:**
   ```bash
   python3 -c "import lxml; print('Success! lxml version:', lxml.__version__)"
   ```

**Solution 2: Install Development Packages (Advanced)**

Only use this if the pre-compiled package is not available:

1. **Close Cygwin terminal**
2. **Run Cygwin Setup:** `setup-x86_64.exe`
3. **Install ALL of these packages:**

   Search for each package and select it:
   - `python39-devel` - Python development headers
   - `libxml2-devel` - XML library headers
   - `libxslt-devel` - XSLT library headers
   - `gcc-core` - C compiler
   - `gcc-g++` - C++ compiler (may be needed)
   - `zlib-devel` - Compression library
   - `make` - Build tool

4. **Complete installation**
5. **Restart Cygwin terminal**
6. **Verify development files are present:**
   ```bash
   ls /usr/include/libxml2/libxml/ 2>/dev/null && echo "libxml2-devel: OK" || echo "libxml2-devel: MISSING"
   ls /usr/include/libxslt/ 2>/dev/null && echo "libxslt-devel: OK" || echo "libxslt-devel: MISSING"
   which gcc && echo "gcc: OK" || echo "gcc: MISSING"
   ```

7. **If all show "OK", try pip again:**
   ```bash
   pip3 install lxml
   ```

**Solution 3: Use apt-cyg (if installed)**

If you have `apt-cyg` package manager:
```bash
apt-cyg install python39-devel libxml2-devel libxslt-devel gcc-core gcc-g++ zlib-devel make
pip3 install lxml
```

**Still failing?**
- Use Solution 1 (pre-compiled binary)
- Or use WSL instead of Cygwin (easier for Python packages)

**Verify installation:**
```bash
python3 -c "import lxml; print('lxml version:', lxml.__version__)"
```

### lxml installation fails on Linux

**Error:** Compilation errors about missing headers

**Solution for Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev zlib1g-dev
pip3 install lxml
```

**Solution for RHEL/CentOS/Fedora:**
```bash
sudo dnf install python3-devel libxml2-devel libxslt-devel zlib-devel
pip3 install lxml
```

### lxml installation fails on macOS

**Error:** Compilation errors or missing Xcode tools

**Solution:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install dependencies via Homebrew
brew install libxml2 libxslt

# Set compiler flags and install
CFLAGS="-I$(brew --prefix libxml2)/include -I$(brew --prefix libxslt)/include" \
LDFLAGS="-L$(brew --prefix libxml2)/lib -L$(brew --prefix libxslt)/lib" \
pip3 install lxml
```

### Permission denied errors

**Error:** `Permission denied` or `Access denied` when installing with pip

**Solution 1 - Install for current user only:**
```bash
pip3 install --user lxml
```

**Solution 2 - Use virtual environment (Recommended):**
```bash
# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install in virtual environment
pip3 install lxml
```

**Solution 3 - Use sudo (Not recommended):**
```bash
# Only if absolutely necessary
sudo pip3 install lxml
```

### Multiple Python versions

**Problem:** Wrong Python version being used

**Solution:**
```bash
# Check which python3 is being used
which python3
python3 --version

# Check all installed Python versions
ls -la /usr/bin/python*

# Use specific version if needed
python3.11 -m pip install lxml

# Or create version-specific alias
alias python3='/usr/bin/python3.11'
```

### WSL-specific issues

**Problem:** `pip3: command not found`

**Solution:**
```bash
# Install pip
sudo apt-get update
sudo apt-get install python3-pip
```

**Problem:** Old pip version causing issues

**Solution:**
```bash
# Upgrade pip
python3 -m pip install --upgrade pip
```

### Verification failures

**Problem:** Installation succeeds but import fails

**Check Python path:**
```bash
python3 -c "import sys; print('\n'.join(sys.path))"
```

**Check installation location:**
```bash
pip3 show lxml
```

**Reinstall cleanly:**
```bash
pip3 uninstall lxml
pip3 install --no-cache-dir lxml
```

### Advanced: Debugging pip compilation issues

If you need to troubleshoot why pip compilation is failing, you can preserve the temporary build folders for inspection.

**Keep build files for debugging:**
```bash
# Preserve temporary build directory
pip3 install --no-clean lxml

# This will leave the build files in a temporary directory when it fails
# The path will be shown in the error output, e.g.:
# /tmp/pip-install-xxxxxx/lxml/
```

**Verbose output for detailed error messages:**
```bash
# Get detailed build output
pip3 install -v lxml          # Verbose
pip3 install -vv lxml         # More verbose
pip3 install -vvv lxml        # Maximum verbosity
```

**Combine both for maximum debugging info:**
```bash
# Keep build files AND show detailed output
pip3 install --no-clean -vvv lxml 2>&1 | tee pip-lxml-debug.log

# This saves all output to pip-lxml-debug.log for analysis
```

**Inspect build files:**
```bash
# After a failed build with --no-clean, you can inspect:
# - C compiler command line arguments
# - Header file locations
# - Linker flags
# - Build scripts

# Example: Check if development headers were found
grep -r "libxml/tree.h" /tmp/pip-install-*/lxml/
```

**Version comparison:**
- **Pre-compiled package (Cygwin):** lxml 4.7.1 (older but stable)
- **pip from source (WSL):** lxml 6.0.2 (latest)

The newer version may have performance improvements and bug fixes. If you need the latest version on Cygwin, you can attempt the compilation approach, but be prepared to install all development packages listed in the [Cygwin troubleshooting section](#lxml-installation-fails-on-cygwin).

## Engine Selection Recommendations

### For Large Files (>100MB)
- **Recommended:** `engine(iterparse)` - Uses lxml, constant memory usage
- **Alternative:** `engine(xmllint_stream)` - Native tool, but limited functionality

### For Small Files (<10MB)
- **Recommended:** `engine(iterparse)` - Still the best option
- **Alternative:** `engine(xmllint_xpath)` - If lxml unavailable

### When Python/lxml Unavailable
- **Use:** `engine(xmllint_stream)` - Basic streaming support
- **Note:** May have limitations with complex tag filtering

## Memory Usage Comparison

Based on Perplexity research and testing:

| Engine | Memory Usage | Large File Support | Installation Complexity |
|--------|--------------|-------------------|------------------------|
| lxml (iterparse) | Constant (~10MB) | ✅ Excellent | Medium (requires pip) |
| xmllint --stream | Variable | ⚠️ Limited | Low (pre-installed) |
| xmlstarlet sel | Loads full file | ❌ Poor | Low (package manager) |

## Dependencies for UnifyWeaver

### Minimal Setup (Basic Functionality)
```bash
# Just xmllint (usually pre-installed on Linux/macOS)
sudo apt-get install libxml2-utils
```

### Recommended Setup (Full Functionality)
```bash
# Install both lxml and xmllint
pip3 install lxml
sudo apt-get install libxml2-utils
```

### Complete Setup (All Engines)
```bash
# Install all tools
pip3 install lxml
sudo apt-get install libxml2-utils xmlstarlet
```

## See Also

- [XML Source Plugin Documentation](./proposals/xml_source/SPECIFICATION.md)
- [Firewall Configuration Guide](./FIREWALL_GUIDE.md)
- [Python Source Plugin](../src/unifyweaver/sources/python_source.pl)

#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# init_testing.sh - Initialize UnifyWeaver testing environment

set -euo pipefail

# Parse command line options
FORCE_WINDOWS=0
CUSTOM_PARENT_DIR=""
CUSTOM_FULL_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --force-windows)
      FORCE_WINDOWS=1
      shift
      ;;
    -d|--dir)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --dir requires a directory argument"
        exit 1
      fi
      CUSTOM_PARENT_DIR="$2"
      shift 2
      ;;
    -p|--path)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --path requires a path argument"
        exit 1
      fi
      CUSTOM_FULL_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -d, --dir <dir>    Parent directory where test_env/ will be created"
      echo "                     Example: --dir /tmp creates /tmp/test_env"
      echo "  -p, --path <path>  Full path to test environment (allows custom name)"
      echo "                     Example: --path /tmp/my_test creates /tmp/my_test"
      echo "  --force-windows    Force use of Windows SWI-Prolog (for testing wrapper logic)"
      echo "  --help             Show this help message"
      echo ""
      echo "Environment Variables:"
      echo "  UNIFYWEAVER_ROOT   Custom target directory for test environment"
      echo "                     (overridden by -p/--path if specified)"
      echo ""
      echo "Examples:"
      echo "  $0                           # Creates test_env in scripts/testing/"
      echo "  $0 -d /tmp                   # Creates /tmp/test_env"
      echo "  $0 -p /tmp/my_custom_test    # Creates /tmp/my_custom_test"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================="
echo "UnifyWeaver Testing Environment Setup"
echo -e "===================================${NC}"

if [[ $FORCE_WINDOWS -eq 1 ]]; then
  echo -e "${YELLOW}NOTE: --force-windows enabled (for testing wrapper upgrade logic)${NC}"
fi

# Get script directory (should be scripts/testing/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Simple config
CFG="$SCRIPT_DIR/.unifyweaver_test.conf"

# Back up the original config file before sourcing, if it exists
BACKUP_CFG_FILE=""
if [[ -f "$SCRIPT_DIR/.unifyweaver.conf" ]]; then
    BACKUP_CFG_FILE="$(mktemp)"
    cp "$SCRIPT_DIR/.unifyweaver.conf" "$BACKUP_CFG_FILE"
fi

if [[ -f "$CFG" ]]; then
  # shellcheck disable=SC1090
  source "$CFG"
fi

FIND_SWIPL_DIR="" #Use relative paths.

# Determine target directory (priority: -p > -d > UNIFYWEAVER_ROOT > default)
if [[ -n "$CUSTOM_FULL_PATH" ]]; then
    TARGET_ROOT="$CUSTOM_FULL_PATH"
    mkdir -p "$TARGET_ROOT"
    echo -e "${YELLOW}Using custom full path: $TARGET_ROOT${NC}"
elif [[ -n "$CUSTOM_PARENT_DIR" ]]; then
    TARGET_ROOT="$CUSTOM_PARENT_DIR/test_env"
    mkdir -p "$TARGET_ROOT"
    echo -e "${YELLOW}Using custom parent directory: $CUSTOM_PARENT_DIR (creating test_env/)${NC}"
elif [[ -n "${UNIFYWEAVER_ROOT:-}" ]]; then
    TARGET_ROOT="$UNIFYWEAVER_ROOT"
    echo -e "${YELLOW}Using UNIFYWEAVER_ROOT environment variable: $TARGET_ROOT${NC}"
else
    TARGET_ROOT="$SCRIPT_DIR/test_env"
    mkdir -p "$TARGET_ROOT"
    echo -e "${YELLOW}Using script directory as target: $TARGET_ROOT${NC}"
fi

# Check if target is empty or non-existent
if [[ -d "$TARGET_ROOT" ]]; then
    # Directory exists, check if it's empty or only contains the init script
    file_count=$(find "$TARGET_ROOT" -mindepth 1 -maxdepth 1 -not -name "init_testing.sh" | wc -l)
    if [[ $file_count -gt 0 ]]; then
        echo -e "${YELLOW}Warning: Target directory is not empty.${NC}"
        echo "Contents:"
        ls -la "$TARGET_ROOT"
        echo ""
        read -p "Continue anyway? This may overwrite files. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# Find the main project root
MAIN_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo -e "\n${YELLOW}Configuration:${NC}"
echo "Main project root: $MAIN_PROJECT_ROOT"
echo "Target directory: $TARGET_ROOT"

# Verify main project structure
#if [[ ! -f "$MAIN_PROJECT_ROOT/src/unifyweaver/core/to_lang.pl" ]]; then
#    echo -e "${RED}Error: Main project structure not found!${NC}"
#    echo "Expected to find: $MAIN_PROJECT_ROOT/src/unifyweaver/core/to_lang.pl"
#    echo "Please run this script from scripts/testing/ in a UnifyWeaver project"
#    exit 1
#fi
#set -x
if [[ $FORCE_WINDOWS -eq 1 ]]; then
  export FORCE_WINDOWS_SWIPL=1
fi
source "$SCRIPT_DIR/find_swi-prolog.sh"
#set +x

# Create directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
dirs=(
    "src/unifyweaver/core"
    "src/unifyweaver/sources"
    "tests/output"
    "templates"
    "scripts"
    "config"
    "examples"
)

for dir in "${dirs[@]}"; do
    mkdir -p "$TARGET_ROOT/$dir"
    echo -e "${GREEN}✓${NC} Created $dir"
done

# Define your directories, including the one you want to exclude
cp_dirs=(
    "src/unifyweaver/core"
    "src/unifyweaver/sources"
    "templates"
    "scripts"
    "config"
    "examples"
)

# (removed hardcoded TARGET_ROOT; keep portable default)

# Create the directories in the target location first
echo -e "\n${YELLOW}Creating destination directories...${NC}"
for dir in "${cp_dirs[@]}"; do
    if mkdir -p "$TARGET_ROOT/$dir" 2>/dev/null; then
        # Silent success
        :
    else
        echo -e "${YELLOW}⚠${NC} Warning: Could not create directory $TARGET_ROOT/$dir"
    fi
done

# Now, copy the contents of each source directory into the destination directories
echo -e "\n${YELLOW}Installing core modules...${NC}"

# Track skipped/failed directories for summary
declare -a SKIPPED_DIRS
declare -a FAILED_DIRS
SKIPPED_DIRS=()
FAILED_DIRS=()

for dir in "${cp_dirs[@]}"; do
    # Skip if source directory doesn't exist
    if [[ ! -d "$MAIN_PROJECT_ROOT/$dir" ]]; then
        echo -e "${YELLOW}⚠${NC} Skipping $dir (not found in main project)"
        SKIPPED_DIRS+=("$dir")
        continue
    fi

    # Use a case statement to handle the special 'scripts' directory
    case "$dir" in
        "scripts")
            # Copy all files from 'scripts' but exclude all 'testing/test_env*' directories
            # The 'rsync' command is ideal for this kind of selective copying
            # The --exclude option lets you skip specific directories or files
            if rsync -av --exclude 'testing/test_env*' "$MAIN_PROJECT_ROOT/$dir/." "$TARGET_ROOT/$dir/" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} Copied contents of scripts, excluding test_env*"
            else
                echo -e "${YELLOW}⚠${NC} Warning: Failed to copy contents of scripts (continuing anyway)"
                FAILED_DIRS+=("$dir")
            fi
            ;;
        *)
            # For all other directories, use the standard 'cp' command
            if cp -r "$MAIN_PROJECT_ROOT/$dir/." "$TARGET_ROOT/$dir/" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} Copied contents of $dir"
            else
                echo -e "${YELLOW}⚠${NC} Warning: Failed to copy contents of $dir (continuing anyway)"
                FAILED_DIRS+=("$dir")
            fi
            ;;
    esac
done

# Optional: move setup_templates.pl if present
if [[ -f "$TARGET_ROOT/templates/setup_templates.pl" ]]; then
  mv "$TARGET_ROOT/templates/setup_templates.pl" "$TARGET_ROOT/setup_templates.pl"
  echo -e "${GREEN}✓${NC} Moved setup_templates.pl to test_env root"
else
  echo -e "${YELLOW}i${NC} No setup_templates.pl found in templates; skipping move"
fi

# Optional: move test_generated_scripts.sh to test_env root if present
if [[ -f "$TARGET_ROOT/examples/test_generated_scripts.sh" ]]; then
  cp "$TARGET_ROOT/examples/test_generated_scripts.sh" "$TARGET_ROOT/test_generated_scripts.sh"
  chmod +x "$TARGET_ROOT/test_generated_scripts.sh"
  echo -e "${GREEN}✓${NC} Copied test_generated_scripts.sh to test_env root"
else
  echo -e "${YELLOW}i${NC} No test_generated_scripts.sh found; skipping"
fi

# Optional: copy init_swipl_env.sh to test_env scripts/ if present
if [[ -f "$SCRIPT_DIR/init_swipl_env.sh" ]]; then
  cp "$SCRIPT_DIR/init_swipl_env.sh" "$TARGET_ROOT/scripts/init_swipl_env.sh"
  chmod +x "$TARGET_ROOT/scripts/init_swipl_env.sh"
  echo -e "${GREEN}✓${NC} Copied init_swipl_env.sh to test_env scripts/"
else
  echo -e "${YELLOW}i${NC} No test_generated_scripts.sh found in examples; skipping"
fi

# Note: backends and pipelines directories are no longer used in v0.0.2

# Copy the main sources.pl facade file
if [[ -f "$MAIN_PROJECT_ROOT/src/unifyweaver/sources.pl" ]]; then
    cp "$MAIN_PROJECT_ROOT/src/unifyweaver/sources.pl" "$TARGET_ROOT/src/unifyweaver/sources.pl"
    echo -e "${GREEN}✓${NC} Copied sources.pl facade module"
else
    echo -e "${YELLOW}i${NC} No sources.pl found in main project"
fi

# Copy test files from the main project
echo -e "\n${YELLOW}Installing test files...${NC}"
if [[ -d "$MAIN_PROJECT_ROOT/tests" ]]; then
    # Copy top-level test files
    cp "$MAIN_PROJECT_ROOT/tests/"*.pl "$TARGET_ROOT/tests/" 2>/dev/null || true
    
    # Copy test files from tests/core/ if it exists
    if [[ -d "$MAIN_PROJECT_ROOT/tests/core" ]]; then
        mkdir -p "$TARGET_ROOT/tests/core"
        cp "$MAIN_PROJECT_ROOT/tests/core/"*.pl "$TARGET_ROOT/tests/core/" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Copied test files from main project (including tests/core/)"
    else
        echo -e "${GREEN}✓${NC} Copied test files from main project"
    fi
else
    echo -e "${YELLOW}i${NC} No tests directory found in main project"
fi

# Create additional test files
cat > "$TARGET_ROOT/tests/test_basic.pl" << 'EOF'
% Basic test to verify compilation works
% Ensure init.pl has run; otherwise consult it
:- (   current_predicate(unifyweaver_initialized/0)
   ->  true
   ;   prolog_load_context(directory, Dir),
       absolute_file_name('../init.pl', Init, [ relative_to(Dir), file_errors(fail) ]),
       consult(Init)
   ).

:- use_module(unifyweaver(core/to_lang)).


% Define simple facts
parent(alice, bob).
parent(bob, charlie).

% Define simple rule  
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

test_basic :-
    format('~n=== Basic Compilation Test ===~n', []),
    
    % Register facts from database
    register_facts_from_db(parent/2),
    
    % Test fact compilation
    format('~nCompiling parent/2 facts to bash...~n', []),
    to_lang(parent/2, bash, [mode-arrays], BashCode),
    format('Generated bash code:~n~s~n', [BashCode]),
    
    % Save to file
    open('tests/output/parent_facts.sh', write, Stream),
    write(Stream, BashCode),
    close(Stream),
    format('~nSaved to tests/output/parent_facts.sh~n', []),
    format('Test completed successfully!~n', []).
EOF

echo -e "${GREEN}✓${NC} Created test_basic.pl"

# Copy init.pl from template
if [[ -f "$MAIN_PROJECT_ROOT/templates/init_template.pl" ]]; then
    cp "$MAIN_PROJECT_ROOT/templates/init_template.pl" "$TARGET_ROOT/init.pl"
    echo -e "${GREEN}✓${NC} Copied init.pl from template"
else
    echo -e "${RED}✗${NC} Template not found: $MAIN_PROJECT_ROOT/templates/init_template.pl"
    echo -e "${YELLOW}i${NC} Falling back to basic init.pl"
    # Fallback: create minimal init.pl
    cat > "$TARGET_ROOT/init.pl" << 'EOF'
% Minimal init.pl (template not found)
:- dynamic user:library_directory/1.
:- dynamic user:file_search_path/2.

unifyweaver_init :-
    prolog_load_context(directory, Here),
    directory_file_path(Here, 'src', AbsSrcDir),
    directory_file_path(AbsSrcDir, 'unifyweaver', AbsUnifyweaverDir),
    asserta(user:library_directory(AbsSrcDir)),
    asserta(user:file_search_path(unifyweaver, AbsUnifyweaverDir)),
    format('[UnifyWeaver] Initialized (minimal mode)~n', []).

:- initialization(unifyweaver_init, now).
EOF
    echo -e "${YELLOW}i${NC} Created minimal init.pl"
fi

# Copy config files
echo -e "\n${YELLOW}Creating configuration files...${NC}"

if [[ -f "$MAIN_PROJECT_ROOT/config/python_paths.txt" ]]; then
    cp "$MAIN_PROJECT_ROOT/config/python_paths.txt" "$TARGET_ROOT/config/"
    echo -e "${GREEN}✓${NC} Copied python_paths.txt from main project"
else
    cat > "$TARGET_ROOT/config/python_paths.txt" << 'EOF'
# Additional Python module paths
# Add your paths here, one per line
# Example: /path/to/agentRAG/src
EOF
    echo -e "${GREEN}✓${NC} Created default python_paths.txt"
fi

# Copy find_swi-prolog.sh helper script early (before any interactive prompts)
echo -e "\n${YELLOW}Installing helper scripts...${NC}"
if [[ -f "$SCRIPT_DIR/find_swi-prolog.sh" ]]; then
    cp -f "$SCRIPT_DIR/find_swi-prolog.sh" "$TARGET_ROOT/find_swi-prolog.sh"
    chmod +x "$TARGET_ROOT/find_swi-prolog.sh"
    echo -e "${GREEN}✓${NC} Copied find_swi-prolog.sh helper script"
else
    echo -e "${RED}✗${NC} Warning: find_swi-prolog.sh not found in $SCRIPT_DIR"
    echo -e "${YELLOW}i${NC} The launcher may not work correctly without this file"
fi

cat > "$TARGET_ROOT/config/assumed_packages.txt" << 'EOF'
# Packages to assume are available without checking
# Add package names here, one per line
# Example: agentRag
# requests
EOF

echo -e "${GREEN}✓${NC} Created assumed_packages.txt"

# Create launcher script

FIND_PREFIX="${FIND_SWIPL_DIR%/}${FIND_SWIPL_DIR:+\/}"

cat > "$TARGET_ROOT/unifyweaver.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail
TARGET_ROOT="."
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
FIND_PREFIX=${FIND_PREFIX}
SCRIPT_DIR="\$(dirname "\$(realpath "\${BASH_SOURCE}")")"
cd "\$SCRIPT_DIR"



source "\$SCRIPT_DIR/find_swi-prolog.sh"

echo "Starting UnifyWeaver testing environment..."
echo "Type 'help.' for available commands"
echo ""
SWIPL_CMD="${SWIPL_CMD:-swipl}"
exec "$SWIPL_CMD" -q -f init.pl
# exec "\$SWIPL_CMD" -q -g "asserta(user:unifyweaver_root('\$SCRIPT_DIR'))" -f init.pl
# Note we don't need the -g option above because in:
# unifyweaver_init
#
# we have:
#
#    retractall(user:unifyweaver_root(_)),
#    assertz(user:unifyweaver_root(Here)),
EOF

chmod +x "$TARGET_ROOT/unifyweaver.sh"
echo -e "${GREEN}✓${NC} Created launcher script"

# Run initial test
echo -e "\n${YELLOW}Running initial test...${NC}"
cd "$TARGET_ROOT"

# Snapshot repo config into the test_env so the launcher is self-contained
if [[ -f "$SCRIPT_DIR/.unifyweaver.conf" ]]; then
  if [[ -n "$FIND_PREFIX" ]]; then
    cp -f "$SCRIPT_DIR/.unifyweaver.conf" "${FIND_PREFIX}.unifyweaver.conf" 2>/dev/null || \
      echo -e "${YELLOW}⚠${NC} Warning: Could not copy .unifyweaver.conf with prefix"
  else
    cp -f "$SCRIPT_DIR/.unifyweaver.conf" ".unifyweaver.conf" 2>/dev/null || \
      echo -e "${YELLOW}⚠${NC} Warning: Could not copy .unifyweaver.conf"
  fi
  if [[ -f ".unifyweaver.conf" ]] || [[ -f "${FIND_PREFIX}.unifyweaver.conf" ]]; then
    echo -e "${GREEN}✓${NC} Copied .unifyweaver.conf to test environment"
  fi
fi

# Ask user about applying settings to the main testing directory
echo ""
read -p "Do you want to apply the new settings to the main testing directory? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✓${NC} Configuration settings updated."
    # If a backup exists, delete it.
    if [[ -n "$BACKUP_CFG_FILE" ]]; then
      rm "$BACKUP_CFG_FILE"
    fi
else
    echo -e "${YELLOW}i${NC} Configuration settings not applied. Restoring original."
    # Restore the backup if it exists, otherwise delete the new file.
    if [[ -n "$BACKUP_CFG_FILE" ]]; then
        mv "$BACKUP_CFG_FILE" "$SCRIPT_DIR/.unifyweaver.conf"
    else
        rm -f "$SCRIPT_DIR/.unifyweaver.conf"
    fi
fi







# Test module loading

SWIPL_CMD=${SWIPL_CMD:-swipl}

if "$SWIPL_CMD" -q -t halt -g "asserta(library_directory('src')), use_module(library(unifyweaver/core/recursive_compiler)), format('Module loaded successfully!~n', [])" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Core modules load correctly"
else
    echo -e "${RED}✗${NC} Module loading failed"
    echo "Debug information:"
    "$SWIPL_CMD" l -q -t halt -g "asserta(library_directory('src')), use_module(library/unifyweaver/core/recursive_compiler))"
fi

echo -e "\n${GREEN}==================================="
echo "Setup Complete!"
echo -e "===================================${NC}"

# Show warnings summary if there were issues
if [[ ${#SKIPPED_DIRS[@]} -gt 0 ]] || [[ ${#FAILED_DIRS[@]} -gt 0 ]]; then
    echo -e "\n${YELLOW}⚠ Warnings Summary:${NC}"
    if [[ ${#SKIPPED_DIRS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Skipped directories (not found):${NC}"
        for dir in "${SKIPPED_DIRS[@]}"; do
            echo "  - $dir"
        done
    fi
    if [[ ${#FAILED_DIRS[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Failed to copy:${NC}"
        for dir in "${FAILED_DIRS[@]}"; do
            echo "  - $dir"
        done
    fi
    echo -e "${YELLOW}Note: The test environment may still be functional.${NC}"
fi

echo -e "\n${YELLOW}Testing Environment Created:${NC}"
echo "Location: $TARGET_ROOT"
echo "Core modules: $(ls -1 "$TARGET_ROOT/src/unifyweaver/core/" 2>/dev/null | wc -l) files"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Run the launcher: ./unifyweaver.sh"
echo "2. In Prolog, type: load_stream. or load_recursive."
echo "3. Or load all: load_all_core."
echo "4. Run tests: test_stream. or test_advanced."

echo -e "\n${YELLOW}File locations:${NC}"
echo "Core modules: src/unifyweaver/core/"
echo "Tests: tests/"
echo "Examples: examples/"
echo "Config: config/"

echo -e "\n${YELLOW}Environment variable usage:${NC}"
echo "UNIFYWEAVER_ROOT=/path/to/testing/dir ./init_testing.sh"

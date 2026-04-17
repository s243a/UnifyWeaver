#!/usr/bin/env bash
# build_scirepl_package.sh — Build a SciREPL package (.zip) from UnifyWeaver
#
# Creates a distributable .zip with:
#   - scirepl.json manifest
#   - prelude.pl (UnifyWeaver environment init for swipl-wasm)
#   - src/unifyweaver/**/*.pl (all Prolog source modules)
#   - education/notebooks/*.ipynb (tutorial notebooks)
#   - examples/*.pl (selected example programs)
#   - init.pl (project initializer)
#
# Usage:
#   ./scripts/build_scirepl_package.sh [output.zip]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="${1:-$PROJECT_ROOT/unifyweaver_scirepl.zip}"
STAGING_DIR=$(mktemp -d)

trap 'rm -rf "$STAGING_DIR"' EXIT

echo "Building SciREPL package from: $PROJECT_ROOT"
echo "Output: $OUTPUT"

# ---- 1. Copy Prolog source files (.pl only, skip runtimes) ----
echo "  Copying src/unifyweaver/ .pl files..."
cd "$PROJECT_ROOT"
find src/unifyweaver -name '*.pl' -type f | while read -r f; do
    dir="$STAGING_DIR/$f"
    mkdir -p "$(dirname "$dir")"
    cp "$f" "$dir"
done
SRC_COUNT=$(find "$STAGING_DIR/src" -name '*.pl' | wc -l)
echo "    $SRC_COUNT .pl files"

# ---- 2. Copy init.pl ----
echo "  Copying init.pl..."
cp init.pl "$STAGING_DIR/init.pl"

# ---- 3. Copy education init.pl(s) ----
echo "  Copying education init files..."
find education -name 'init.pl' -type f 2>/dev/null | while read -r f; do
    mkdir -p "$STAGING_DIR/$(dirname "$f")"
    cp "$f" "$STAGING_DIR/$f"
done

# ---- 4. Copy education notebooks ----
echo "  Copying education notebooks..."
mkdir -p "$STAGING_DIR/education/notebooks"
for nb in education/notebooks/*.ipynb; do
    [ -f "$nb" ] && cp "$nb" "$STAGING_DIR/$nb"
done
NB_COUNT=$(find "$STAGING_DIR/education/notebooks" -name '*.ipynb' 2>/dev/null | wc -l)
echo "    $NB_COUNT notebooks"

# ---- 5. Copy selected examples ----
echo "  Copying examples/*.pl..."
mkdir -p "$STAGING_DIR/examples"
for f in examples/*.pl; do
    [ -f "$f" ] && cp "$f" "$STAGING_DIR/$f"
done
EX_COUNT=$(find "$STAGING_DIR/examples" -name '*.pl' 2>/dev/null | wc -l)
echo "    $EX_COUNT examples"

# ---- 5b. Copy workbooks listed in PACKAGE_WORKBOOKS ----
# Each entry is: source_path
# Paths are relative to PROJECT_ROOT.
PACKAGE_WORKBOOKS=(
    "examples/sci-repl/prototype/www/workbooks/prolog-generates-r.srwb"
)

echo "  Copying workbooks..."
mkdir -p "$STAGING_DIR/workbooks"
WB_COUNT=0
for wb in "${PACKAGE_WORKBOOKS[@]}"; do
    if [ -f "$wb" ]; then
        cp "$wb" "$STAGING_DIR/workbooks/$(basename "$wb")"
        echo "    + $(basename "$wb")"
        WB_COUNT=$((WB_COUNT + 1))
    else
        echo "    ! MISSING: $wb"
    fi
done
echo "    $WB_COUNT workbooks"

# ---- 5c. Copy template files (mustache, tmpl.sh) ----
echo "  Copying templates/..."
if [ -d "templates" ]; then
    find templates -type f \( -name '*.mustache' -o -name '*.tmpl.sh' -o -name '*.txt' \) | while read -r f; do
        mkdir -p "$STAGING_DIR/$(dirname "$f")"
        cp "$f" "$STAGING_DIR/$f"
    done
    TMPL_COUNT=$(find "$STAGING_DIR/templates" -type f 2>/dev/null | wc -l)
    echo "    $TMPL_COUNT template files"
else
    echo "    ! templates/ directory not found"
    TMPL_COUNT=0
fi

# ---- 6. Generate prelude.pl ----
echo "  Generating prelude.pl..."
cat > "$STAGING_DIR/prelude.pl" << 'PRELUDE_EOF'
%% prelude.pl — UnifyWeaver environment for SciREPL (swipl-wasm)
%%
%% This file initializes the UnifyWeaver module system inside the
%% browser-based SWI-Prolog WASM kernel. It sets up file_search_path/2
%% entries so that use_module(unifyweaver(...)) calls resolve correctly
%% against the virtual filesystem.
%%
%% Auto-loaded by SciREPL when a UnifyWeaver package is imported.

:- module(uw_prelude, [
    uw_version/1,
    uw_root/1,
    show/1,
    show_all/1
]).

:- dynamic user:file_search_path/2.
:- dynamic user:unifyweaver_root/1.

%% uw_version(-Version)
%  Current UnifyWeaver version string.
uw_version('0.5.0-wasm').

%% uw_root(-Root)
%  The VFS root where UnifyWeaver files are mounted.
uw_root('/user').

%% Initialize search paths for the VFS layout.
%%
%% VFS structure (set up by PackageLoader):
%%   /user/src/unifyweaver/core/...
%%   /user/src/unifyweaver/targets/...
%%   /user/src/unifyweaver/bindings/...
%%   /user/examples/...
%%   /user/education/...
%%   /user/init.pl
%%   /user/prelude.pl

:- (   user:file_search_path(unifyweaver, _)
   ->  true
   ;   assertz(user:file_search_path(unifyweaver, '/user/src/unifyweaver'))
   ).

:- (   user:file_search_path(uw_core, _)
   ->  true
   ;   assertz(user:file_search_path(uw_core, '/user/src/unifyweaver/core'))
   ).

:- (   user:file_search_path(uw_targets, _)
   ->  true
   ;   assertz(user:file_search_path(uw_targets, '/user/src/unifyweaver/targets'))
   ).

:- (   user:file_search_path(uw_bindings, _)
   ->  true
   ;   assertz(user:file_search_path(uw_bindings, '/user/src/unifyweaver/bindings'))
   ).

:- (   user:file_search_path(uw_sources, _)
   ->  true
   ;   assertz(user:file_search_path(uw_sources, '/user/src/unifyweaver/sources'))
   ).

:- (   user:file_search_path(uw_glue, _)
   ->  true
   ;   assertz(user:file_search_path(uw_glue, '/user/src/unifyweaver/glue'))
   ).

:- (   user:file_search_path(uw_examples, _)
   ->  true
   ;   assertz(user:file_search_path(uw_examples, '/user/examples'))
   ).

:- (   user:file_search_path(uw_education, _)
   ->  true
   ;   assertz(user:file_search_path(uw_education, '/user/education'))
   ).

%% show(+Term)
%  Pretty-print a term to stdout.
show(X) :- print(X), nl.

%% show_all(+List)
%  Pretty-print each element of a list.
show_all([]).
show_all([H|T]) :- show(H), show_all(T).

%% between/3 polyfill (if not already available)
:- if(\+ current_predicate(between/3)).
between(Low, High, Low) :- Low =< High.
between(Low, High, X) :- Low < High, Low1 is Low + 1, between(Low1, High, X).
:- endif.
PRELUDE_EOF

# ---- 7. Generate scirepl.json manifest ----
echo "  Generating scirepl.json..."

# Build notebook entries as a JSON array using a temp file
NB_TMP="$STAGING_DIR/_nb_entries.txt"
: > "$NB_TMP"
for nb in "$STAGING_DIR"/education/notebooks/*.ipynb; do
    [ -f "$nb" ] || continue
    basename_nb=$(basename "$nb")
    name=$(echo "$basename_nb" | sed 's/\.ipynb$//' | sed 's/_/ /g' | sed 's/^[0-9]* //')
    echo "    { \"file\": \"education/notebooks/${basename_nb}\", \"name\": \"${name}\", \"kernel\": \"prolog\" }" >> "$NB_TMP"
done
# Add workbook entries (.srwb files from workbooks/)
for wb in "$STAGING_DIR"/workbooks/*.srwb; do
    [ -f "$wb" ] || continue
    basename_wb=$(basename "$wb")
    name=$(echo "$basename_wb" | sed 's/\.srwb$//' | sed 's/-/ /g' | sed 's/\b\(.\)/\u\1/g')
    echo "    { \"file\": \"workbooks/${basename_wb}\", \"name\": \"${name}\", \"type\": \"srwb\" }" >> "$NB_TMP"
done
# Join lines with commas (all but last get a trailing comma)
NB_ENTRIES=$(sed '$!s/$/,/' "$NB_TMP")
rm -f "$NB_TMP"

# Build file entries for VFS mounting
cat > "$STAGING_DIR/scirepl.json" << MANIFEST_EOF
{
  "format_version": "1.0",
  "name": "UnifyWeaver",
  "version": "0.5.0",
  "description": "A Prolog transpiler that compiles logic programs into multi-language data pipelines. Includes source modules, tutorials, and examples.",
  "notebooks": [
$NB_ENTRIES
  ],
  "files": [
    { "src": "prelude.pl", "dest": "/user/uw_prelude.pl" },
    { "src": "init.pl", "dest": "/user/init.pl" },
    { "src": "src/", "dest": "/user/src/" },
    { "src": "examples/", "dest": "/user/examples/" },
    { "src": "education/", "dest": "/user/education/" },
    { "src": "workbooks/", "dest": "/user/workbooks/" },
    { "src": "templates/", "dest": "/user/templates/" }
  ],
  "search_paths": [
    { "alias": "unifyweaver", "dir": "/user/src/unifyweaver" },
    { "alias": "uw_core", "dir": "/user/src/unifyweaver/core" },
    { "alias": "uw_targets", "dir": "/user/src/unifyweaver/targets" },
    { "alias": "uw_bindings", "dir": "/user/src/unifyweaver/bindings" },
    { "alias": "uw_sources", "dir": "/user/src/unifyweaver/sources" },
    { "alias": "uw_glue", "dir": "/user/src/unifyweaver/glue" },
    { "alias": "uw_examples", "dir": "/user/examples" },
    { "alias": "uw_education", "dir": "/user/education" }
  ]
}
MANIFEST_EOF

# ---- 8. Create .zip ----
echo "  Creating zip archive..."
cd "$STAGING_DIR"
rm -f "$OUTPUT"
zip -r -q "$OUTPUT" .

SIZE=$(du -sh "$OUTPUT" | cut -f1)
FILE_COUNT=$(find "$STAGING_DIR" -type f | wc -l)

echo ""
echo "Done! Package: $OUTPUT ($SIZE, $FILE_COUNT files)"
echo ""
echo "Contents:"
echo "  - scirepl.json (manifest)"
echo "  - prelude.pl (WASM environment init)"
echo "  - init.pl (project initializer)"
echo "  - src/unifyweaver/ ($SRC_COUNT .pl modules)"
echo "  - education/notebooks/ ($NB_COUNT notebooks)"
echo "  - examples/ ($EX_COUNT examples)"
echo "  - workbooks/ ($WB_COUNT workbooks)"
echo "  - templates/ ($TMPL_COUNT template files)"
echo ""
echo "To use: Import in SciREPL via Menu > Import Package"

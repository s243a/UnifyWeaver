---
file_type: UnifyWeaver Example Library
---
# Playbook Examples: CSV Data Processing

This file contains executable records for UnifyWeaver playbooks related to CSV data processing.

## Record: unifyweaver.execution.csv_data_source

This record demonstrates how to use UnifyWeaver's CSV source to read and process user data.

> [!example-record]
> id: unifyweaver.execution.csv_data_source
> name: unifyweaver.execution.csv_data_source
> description: Demonstrates CSV source with data filtering to find user by name

```bash
#!/bin/bash
#
# CSV Data Source Example - Demonstrates CSV source with filtering
# This script is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
PROLOG_SCRIPT="$TMP_FOLDER/csv_example.pl"
OUTPUT_SCRIPT="$TMP_FOLDER/get_user_age.sh"

# --- Step 1: Create Prolog Script ---
echo "Creating Prolog script..."
mkdir -p "$TMP_FOLDER"
cat > "$PROLOG_SCRIPT" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Define a CSV source that reads user data
:- source(csv, users, [
    csv_file('test_data/test_users.csv'),
    has_header(true)
]).

% Define a predicate to get user age by name
get_user_age(Name, Age) :-
    users(_, Name, Age).

run_example :-
    % Compile the predicate to bash
    compile_dynamic_source(users/3, [], BashCode),
    % Write to file
    open('tmp/users.sh', write, Stream1),
    write(Stream1, BashCode),
    close(Stream1),
    format('~nCompiled CSV source to tmp/users.sh~n', []),

    % Note: get_user_age/2 would need recursive compilation
    % For this simple example, we just demonstrate the CSV source compilation
    format('~nTo use: source tmp/users.sh && users~n', []).
PROLOG_EOF

# --- Step 2: Execute Prolog to Generate Bash Script ---
echo "Compiling CSV source to bash..."
swipl -g "consult('$PROLOG_SCRIPT'), run_example, halt"

# --- Step 3: Test Generated Script ---
echo ""
echo "Testing generated bash script..."
if [[ -f "tmp/users.sh" ]]; then
    echo "Loading users function..."
    source tmp/users.sh
    echo ""
    echo "Calling users() to get all records:"
    users
    echo ""
    echo "Success: CSV source compiled and executed"
else
    echo "Error: Expected output script not found: tmp/users.sh"
    exit 1
fi
```

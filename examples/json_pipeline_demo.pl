:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% json_pipeline_demo.pl - JSON source pipeline demonstration  
% Shows JSON parsing with jq filters and transformations

:- initialization(main, main).

%% Load infrastructure
:- use_module('src/unifyweaver/sources').
:- load_files('src/unifyweaver/sources/json_source', [imports([])]).
:- load_files('src/unifyweaver/core/dynamic_source_compiler', [imports([])]).

%% Define JSON sources

% JSON source 1: Extract user names from JSON file
:- source(json, extract_users, [
    json_file('output/sample.json'),
    jq_filter('.users[] | [.name, .role] | @tsv'),
    raw_output(true)
]).

% JSON source 2: Parse JSON from stdin and extract specific fields
:- source(json, parse_data, [
    jq_filter('.[] | select(.active == true) | .name'),
    json_stdin(true),
    raw_output(true)
]).

%% ============================================
%% MAIN PIPELINE
%% ============================================

main :-
    format('📋 JSON Source Pipeline Demo~n', []),
    format('=================================~n~n', []),
    
    % Step 1: Create sample JSON
    format('📝 Step 1: Creating sample JSON data...~n', []),
    create_sample_json,
    
    % Step 2: Compile JSON sources
    format('~n🔨 Step 2: Compiling JSON sources...~n', []),
    compile_json_sources,
    
    % Step 3: Execute pipeline
    format('~n🚀 Step 3: Processing JSON data...~n', []),
    execute_json_pipeline,
    
    format('~n✅ JSON Pipeline Complete!~n~n', []),
    format('Generated files:~n', []),
    format('  - output/sample.json (sample data)~n', []),
    format('  - output/extract_users.sh (file parser)~n', []),
    format('  - output/parse_data.sh (stdin parser)~n', []),
    !.

main :-
    format('~n❌ Pipeline failed!~n', []),
    halt(1).

%% ============================================
%% PIPELINE STEPS
%% ============================================

create_sample_json :-
    (exists_directory('output') -> true ; make_directory('output')),
    
    % Create sample JSON file
    JSON = '{
  "users": [
    {"name": "Alice", "role": "Developer", "active": true},
    {"name": "Bob", "role": "Designer", "active": true},
    {"name": "Charlie", "role": "Manager", "active": false},
    {"name": "Diana", "role": "Analyst", "active": true}
  ],
  "metadata": {
    "version": "1.0",
    "timestamp": "2025-10-16"
  }
}',
    
    open('output/sample.json', write, Stream),
    write(Stream, JSON),
    close(Stream),
    format('   ✓ Created output/sample.json~n', []).

compile_json_sources :-
    % Compile extract_users
    format('   Compiling extract_users/2...~n', []),
    dynamic_source_compiler:compile_dynamic_source(extract_users/2, [], BashCode1),
    open('output/extract_users.sh', write, S1),
    format(S1, '~s', [BashCode1]),
    close(S1),
    shell('chmod +x output/extract_users.sh', _),
    
    % Compile parse_data
    format('   Compiling parse_data/2...~n', []),
    dynamic_source_compiler:compile_dynamic_source(parse_data/2, [], BashCode2),
    open('output/parse_data.sh', write, S2),
    format(S2, '~s', [BashCode2]),
    close(S2),
    shell('chmod +x output/parse_data.sh', _),
    
    format('   ✓ Generated JSON source scripts~n', []).

execute_json_pipeline :-
    % Test 1: Extract users from file
    format('   Extracting users from JSON file...~n', []),
    shell('bash output/extract_users.sh 2>/dev/null', Status1),
    (Status1 = 0 -> 
        format('   ✓ Users extracted (shown above)~n', [])
    ; 
        format('   ℹ jq not available (install with: apt install jq)~n', [])
    ),
    
    % Test 2: Filter active users via stdin
    format('~n   Filtering active users from stdin...~n', []),
    shell('cat output/sample.json | bash output/parse_data.sh 2>/dev/null', Status2),
    (Status2 = 0 -> 
        format('   ✓ Active users filtered (shown above)~n', [])
    ; 
        format('   ℹ jq not available (install with: apt install jq)~n', [])
    ),
    
    % Show jq availability
    format('~n   Checking jq installation:~n', []),
    shell('which jq >/dev/null 2>&1 && echo "   ✓ jq is installed" || echo "   ✗ jq not found (optional for JSON processing)"', _).

%% ============================================
%% USAGE
%% ============================================

/*
To run this JSON pipeline demo:

cd scripts/testing/test_env5
swipl -g main -t halt examples/json_pipeline_demo.pl

This demonstrates:
✅ JSON file parsing with jq
✅ JSON filtering and transformation
✅ TSV output format (@tsv)
✅ Stdin JSON processing
✅ Complex jq filters (select, field extraction)

The pipeline:
1. Creates sample JSON with users array
2. Compiles JSON sources to bash
3. Extracts user names and roles to TSV
4. Filters for active users only
5. Shows results

Note: Requires jq (JSON processor)
Install with: sudo apt install jq
If unavailable, demo explains what would happen.
*/

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% pipeline_demo.pl - Complete working pipeline demonstration
% Shows data sources being compiled to bash and executed

:- initialization(main, main).

%% Load the data sources infrastructure
:- use_module('src/unifyweaver/sources').
:- load_files('src/unifyweaver/sources/csv_source', [imports([])]).
:- load_files('src/unifyweaver/core/dynamic_source_compiler', [imports([])]).

%% Define data sources
:- source(csv, users, [
    csv_file('examples/demo_users.csv'),
    has_header(true)
]).

%% ============================================
%% MAIN PIPELINE EXECUTION
%% ============================================

main :-
    format('🎯 UnifyWeaver Data Sources Pipeline Demo~n', []),
    format('==============================================~n~n', []),
    
    % Step 1: Create sample data
    format('📝 Step 1: Creating sample CSV data...~n', []),
    create_sample_data,
    
    % Step 2: Compile source to bash
    format('~n🔨 Step 2: Compiling CSV source to bash...~n', []),
    compile_sources_to_bash,
    
    % Step 3: Execute bash pipeline
    format('~n🚀 Step 3: Executing bash pipeline...~n', []),
    execute_pipeline,
    
    % Step 4: Verify results
    format('~n✅ Pipeline Demo Complete!~n~n', []),
    format('Generated files:~n', []),
    format('  - examples/demo_users.csv (input data)~n', []),
    format('  - output/users.sh (compiled bash)~n', []),
    format('  - output/pipeline_results.txt (pipeline output)~n', []),
    !.

main :-
    format('~n❌ Pipeline demo failed!~n', []),
    halt(1).

%% ============================================
%% PIPELINE STEPS
%% ============================================

%% Create sample CSV data
create_sample_data :-
    (exists_directory('examples') -> true ; make_directory('examples')),
    (exists_directory('output') -> true ; make_directory('output')),
    
    open('examples/demo_users.csv', write, Stream),
    write(Stream, 'id,name,role,department\n'),
    write(Stream, '1,Alice,Developer,Engineering\n'),
    write(Stream, '2,Bob,Designer,Design\n'),
    write(Stream, '3,Charlie,Manager,Operations\n'),
    write(Stream, '4,Diana,Analyst,Data\n'),
    close(Stream),
    format('   ✓ Created examples/demo_users.csv with 4 users~n', []).

%% Compile sources to bash
compile_sources_to_bash :-
    % Compile the users source
    format('   Compiling users/4...~n', []),
    dynamic_source_compiler:compile_dynamic_source(users/4, [], BashCode),
    
    % Write to file
    open('output/users.sh', write, Stream),
    write(Stream, BashCode),
    close(Stream),
    
    % Make executable
    shell('chmod +x output/users.sh', _),
    
    format('   ✓ Generated output/users.sh~n', []).

%% Execute the pipeline
execute_pipeline :-
    % Source the bash functions
    PipelineScript = '#!/bin/bash\n\
# Load the generated source functions\n\
source output/users.sh\n\
\n\
echo "=== Pipeline Execution Results ===" > output/pipeline_results.txt\n\
echo "" >> output/pipeline_results.txt\n\
\n\
# Example 1: Stream all users\n\
echo "1. All users:" >> output/pipeline_results.txt\n\
users_stream | while IFS=: read id name role department; do\n\
    echo "  - $name ($role) in $department" >> output/pipeline_results.txt\n\
done\n\
\n\
# Example 2: Filter for developers\n\
echo "" >> output/pipeline_results.txt\n\
echo "2. Developers only:" >> output/pipeline_results.txt\n\
users_stream | awk -F: \'$3 == "Developer"\' | while IFS=: read id name role department; do\n\
    echo "  - $name" >> output/pipeline_results.txt\n\
done\n\
\n\
# Example 3: Count by department\n\
echo "" >> output/pipeline_results.txt\n\
echo "3. Users by department:" >> output/pipeline_results.txt\n\
users_stream | awk -F: \'{dept[$4]++} END {for (d in dept) print "  - " d ": " dept[d]}\' >> output/pipeline_results.txt\n\
\n\
echo "" >> output/pipeline_results.txt\n\
echo "Pipeline completed successfully!" >> output/pipeline_results.txt\n',
    
    % Write pipeline script
    open('output/run_pipeline.sh', write, Stream),
    write(Stream, PipelineScript),
    close(Stream),
    
    % Execute
    shell('chmod +x output/run_pipeline.sh', _),
    shell('bash output/run_pipeline.sh', Status),
    
    (   Status = 0
    ->  format('   ✓ Pipeline executed successfully~n', []),
        format('~n📊 Pipeline Results:~n', []),
        shell('cat output/pipeline_results.txt', _)
    ;   format('   ✗ Pipeline execution failed with status ~w~n', [Status])
    ).

%% ============================================
%% USAGE
%% ============================================

/*
To run this complete pipeline demo:

1. From command line:
   cd scripts/testing/test_env5
   swipl -g main -t halt examples/pipeline_demo.pl

2. From UnifyWeaver environment:
   ./unifyweaver.sh
   ?- [examples/pipeline_demo].
   ?- main.

This demonstrates:
✅ CSV data source definition
✅ Compilation to bash
✅ Bash function generation
✅ Pipeline execution
✅ Data transformation
✅ Real output generation

The pipeline:
1. Creates CSV with user data
2. Compiles users/4 source to bash
3. Executes bash pipeline that:
   - Streams all users
   - Filters developers
   - Counts by department
4. Writes results to output/pipeline_results.txt
*/

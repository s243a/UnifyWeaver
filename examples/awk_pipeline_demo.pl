:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% awk_pipeline_demo.pl - AWK source pipeline demonstration
% Shows AWK-based data processing and transformation

:- initialization(main, main).

%% Load infrastructure
:- use_module('src/unifyweaver/sources').
:- load_files('src/unifyweaver/sources/awk_source', [imports([])]).
:- load_files('src/unifyweaver/core/dynamic_source_compiler', [imports([])]).

%% Define AWK sources

% AWK source 1: Parse log file and extract errors
:- source(awk, extract_errors, [
    awk_command('$3 == "ERROR" { print $1, $2, $4 }'),
    delimiter(' ')
]).

% AWK source 2: Calculate statistics from data
:- source(awk, calc_stats, [
    awk_command('{ sum += $2; count++ } END { print "Total:", sum, "Average:", sum/count }'),
    delimiter(':')
]).

% AWK source 3: Transform CSV to different format
:- source(awk, transform_data, [
    awk_command('BEGIN { FS=","; OFS=":" } NR > 1 { print $1, $2, $3 }')
]).

%% ============================================
%% MAIN PIPELINE
%% ============================================

main :-
    format('🔧 AWK Source Pipeline Demo~n', []),
    format('================================~n~n', []),
    
    % Step 1: Create sample data
    format('📝 Step 1: Creating sample log and data files...~n', []),
    create_sample_data,
    
    % Step 2: Compile AWK sources
    format('~n🔨 Step 2: Compiling AWK sources...~n', []),
    compile_awk_sources,
    
    % Step 3: Execute pipeline
    format('~n🚀 Step 3: Processing data with AWK...~n', []),
    execute_awk_pipeline,
    
    format('~n✅ AWK Pipeline Complete!~n~n', []),
    format('Generated files:~n', []),
    format('  - output/sample_log.txt (log data)~n', []),
    format('  - output/sample_stats.txt (numeric data)~n', []),
    format('  - output/extract_errors.sh (error extractor)~n', []),
    format('  - output/calc_stats.sh (stats calculator)~n', []),
    format('  - output/transform_data.sh (data transformer)~n', []),
    !.

main :-
    format('~n❌ Pipeline failed!~n', []),
    halt(1).

%% ============================================
%% PIPELINE STEPS
%% ============================================

create_sample_data :-
    (exists_directory('output') -> true ; make_directory('output')),
    
    % Create sample log file
    open('output/sample_log.txt', write, LogStream),
    write(LogStream, '2025-10-16 10:00:01 INFO Application started\n'),
    write(LogStream, '2025-10-16 10:00:15 ERROR Database connection failed\n'),
    write(LogStream, '2025-10-16 10:00:30 INFO Processing request\n'),
    write(LogStream, '2025-10-16 10:00:45 ERROR Timeout occurred\n'),
    write(LogStream, '2025-10-16 10:01:00 INFO Request completed\n'),
    write(LogStream, '2025-10-16 10:01:15 ERROR Invalid input\n'),
    close(LogStream),
    format('   ✓ Created output/sample_log.txt~n', []),
    
    % Create sample stats data
    open('output/sample_stats.txt', write, StatsStream),
    write(StatsStream, 'user1:95\n'),
    write(StatsStream, 'user2:87\n'),
    write(StatsStream, 'user3:92\n'),
    write(StatsStream, 'user4:98\n'),
    write(StatsStream, 'user5:85\n'),
    close(StatsStream),
    format('   ✓ Created output/sample_stats.txt~n', []),
    
    % Create sample CSV
    open('output/sample_csv.txt', write, CsvStream),
    write(CsvStream, 'id,name,score\n'),
    write(CsvStream, '1,Alice,95\n'),
    write(CsvStream, '2,Bob,87\n'),
    write(CsvStream, '3,Charlie,92\n'),
    close(CsvStream),
    format('   ✓ Created output/sample_csv.txt~n', []).

compile_awk_sources :-
    % Compile extract_errors
    format('   Compiling extract_errors/3...~n', []),
    dynamic_source_compiler:compile_dynamic_source(extract_errors/3, [], BashCode1),
    open('output/extract_errors.sh', write, S1),
    write(S1, BashCode1),
    close(S1),
    shell('chmod +x output/extract_errors.sh', _),
    
    % Compile calc_stats
    format('   Compiling calc_stats/1...~n', []),
    dynamic_source_compiler:compile_dynamic_source(calc_stats/1, [], BashCode2),
    open('output/calc_stats.sh', write, S2),
    write(S2, BashCode2),
    close(S2),
    shell('chmod +x output/calc_stats.sh', _),
    
    % Compile transform_data
    format('   Compiling transform_data/3...~n', []),
    dynamic_source_compiler:compile_dynamic_source(transform_data/3, [], BashCode3),
    open('output/transform_data.sh', write, S3),
    write(S3, BashCode3),
    close(S3),
    shell('chmod +x output/transform_data.sh', _),
    
    format('   ✓ Generated AWK source scripts~n', []).

execute_awk_pipeline :-
    % Test 1: Extract errors from log
    format('   Extracting errors from log file...~n', []),
    shell('cat output/sample_log.txt | bash output/extract_errors.sh', Status1),
    (Status1 = 0 -> 
        format('   ✓ Errors extracted (shown above)~n', [])
    ; 
        format('   ✗ Extraction failed~n', [])
    ),
    
    % Test 2: Calculate statistics
    format('~n   Calculating statistics from data...~n', []),
    shell('cat output/sample_stats.txt | bash output/calc_stats.sh', Status2),
    (Status2 = 0 -> 
        format('   ✓ Statistics calculated (shown above)~n', [])
    ; 
        format('   ✗ Calculation failed~n', [])
    ),
    
    % Test 3: Transform CSV data
    format('~n   Transforming CSV data...~n', []),
    shell('cat output/sample_csv.txt | bash output/transform_data.sh', Status3),
    (Status3 = 0 -> 
        format('   ✓ Data transformed (shown above)~n', [])
    ; 
        format('   ✗ Transformation failed~n', [])
    ),
    
    % Show AWK availability
    format('~n   Checking AWK installation:~n', []),
    shell('which awk >/dev/null 2>&1 && echo "   ✓ AWK is installed" || echo "   ✗ AWK not found"', _).

%% ============================================
%% USAGE
%% ============================================

/*
To run this AWK pipeline demo:

cd scripts/testing/test_env5
swipl -g main -t halt examples/awk_pipeline_demo.pl

This demonstrates:
✅ AWK command execution
✅ Log file parsing and filtering
✅ Statistical calculations
✅ Data transformation
✅ CSV processing with field separators

The pipeline:
1. Creates sample log and data files
2. Compiles AWK sources to bash
3. Extracts ERROR entries from logs
4. Calculates sum and average from data
5. Transforms CSV to colon-separated format

AWK Operations:
- Pattern matching ($3 == "ERROR")
- Field extraction ($1, $2, $4)
- Accumulation (sum, count)
- BEGIN/END blocks
- Field separators (FS, OFS)

Note: AWK is standard on Unix/Linux systems
*/

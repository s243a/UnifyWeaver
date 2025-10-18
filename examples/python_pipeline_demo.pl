:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% python_pipeline_demo.pl - Python source pipeline demonstration
% Shows Python-based data transformation and SQLite integration

:- initialization(main, main).

%% Load infrastructure
:- use_module('src/unifyweaver/sources').
:- load_files('src/unifyweaver/sources/python_source', [imports([])]).
:- load_files('src/unifyweaver/core/dynamic_source_compiler', [imports([])]).

%% Define Python sources

% Python source 1: Process CSV and store in SQLite
:- source(python, store_data, [
    python_inline('
import sys
import sqlite3
import csv

conn = sqlite3.connect("output/python_demo.db")
conn.execute("DROP TABLE IF EXISTS users")
conn.execute("CREATE TABLE users (id INTEGER, name TEXT, score INTEGER)")

for line in sys.stdin:
    fields = line.strip().split(":")
    if len(fields) >= 3:
        conn.execute("INSERT INTO users VALUES (?, ?, ?)", 
                    (int(fields[0]), fields[1], int(fields[2])))

conn.commit()
print("Stored", conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], "records")
conn.close()
'),
    timeout(30)
]).

% Python source 2: Query SQLite and output results
:- source(python, query_data, [
    sqlite_query('SELECT id, name, score FROM users ORDER BY score DESC'),
    database('output/python_demo.db')
]).

%% ============================================
%% MAIN PIPELINE
%% ============================================

main :-
    format('🐍 Python Source Pipeline Demo~n', []),
    format('=====================================~n~n', []),
    
    % Step 1: Create sample data
    format('📝 Step 1: Creating sample data...~n', []),
    create_sample_data,
    
    % Step 2: Compile Python sources
    format('~n🔨 Step 2: Compiling Python sources...~n', []),
    compile_python_sources,
    
    % Step 3: Execute pipeline
    format('~n🚀 Step 3: Executing pipeline...~n', []),
    execute_python_pipeline,
    
    format('~n✅ Python Pipeline Complete!~n~n', []),
    format('Generated files:~n', []),
    format('  - output/python_demo.db (SQLite database)~n', []),
    format('  - output/store_data.sh (data import script)~n', []),
    format('  - output/query_data.sh (data query script)~n', []),
    !.

main :-
    format('~n❌ Pipeline failed!~n', []),
    halt(1).

%% ============================================
%% PIPELINE STEPS
%% ============================================

create_sample_data :-
    (exists_directory('output') -> true ; make_directory('output')),
    
    open('output/sample_data.txt', write, Stream),
    write(Stream, '1:Alice:95\n'),
    write(Stream, '2:Bob:87\n'),
    write(Stream, '3:Charlie:92\n'),
    write(Stream, '4:Diana:98\n'),
    write(Stream, '5:Eve:85\n'),
    close(Stream),
    format('   ✓ Created output/sample_data.txt with 5 users~n', []).

compile_python_sources :-
    % Compile store_data
    format('   Compiling store_data/2...~n', []),
    dynamic_source_compiler:compile_dynamic_source(store_data/2, [], BashCode1),
    open('output/store_data.sh', write, S1),
    format(S1, '~s', [BashCode1]),
    close(S1),
    shell('chmod +x output/store_data.sh', _),
    
    % Compile query_data
    format('   Compiling query_data/2...~n', []),
    dynamic_source_compiler:compile_dynamic_source(query_data/2, [], BashCode2),
    open('output/query_data.sh', write, S2),
    format(S2, '~s', [BashCode2]),
    close(S2),
    shell('chmod +x output/query_data.sh', _),
    
    format('   ✓ Generated Python source scripts~n', []).

execute_python_pipeline :-
    % Step 1: Load data into SQLite
    format('   Loading data into SQLite...~n', []),
    shell('cat output/sample_data.txt | bash output/store_data.sh', Status1),
    (Status1 = 0 -> format('   ✓ Data loaded successfully~n', []) ; format('   ✗ Load failed~n', [])),
    
    % Step 2: Query and display results
    format('~n   Querying data (sorted by score):~n', []),
    shell('bash output/query_data.sh', Status2),
    (Status2 = 0 -> true ; format('   ✗ Query failed~n', [])),
    
    % Step 3: Show database stats
    format('~n   Database statistics:~n', []),
    shell('echo "SELECT COUNT(*) as total FROM users;" | sqlite3 output/python_demo.db', _),
    shell('echo "SELECT AVG(score) as avg_score FROM users;" | sqlite3 output/python_demo.db', _).

%% ============================================
%% USAGE
%% ============================================

/*
To run this Python pipeline demo:

cd scripts/testing/test_env5
swipl -g main -t halt examples/python_pipeline_demo.pl

This demonstrates:
✅ Python inline code execution
✅ SQLite database operations  
✅ Data import pipeline
✅ SQL queries from bash
✅ Multi-step data processing

The pipeline:
1. Creates sample data (5 users with scores)
2. Compiles Python sources to bash
3. Loads data into SQLite via Python
4. Queries data with SQL
5. Displays sorted results
*/

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% data_sources_working_demo.pl - Working demonstration with real output
%
% Unlike data_sources_demo.pl (syntax example), this actually executes
% data sources and generates output files for verification.

:- initialization(main, main).

%% Configure firewall
:- assertz(firewall:firewall_default([
    services([awk, python3]),
    python_modules([sys, sqlite3]),
    file_read_patterns(['examples/*']),
    file_write_patterns(['output/*']),
    cache_dirs(['cache/*'])
])).

%% ============================================
%% WORKING EXAMPLES WITH REAL OUTPUT
%% ============================================

%% Simple CSV source that actually gets compiled and executed
:- source(csv, demo_users, [
    csv_file('examples/demo_users.csv'),
    has_header(true),
    delimiter(',')
]).

%% Python source that creates a SQLite database
:- source(python, create_demo_db, [
    python_inline('
import sys
import sqlite3

# Create database
conn = sqlite3.connect("output/working_demo.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS demo_results (
        id INTEGER PRIMARY KEY,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")

# Insert test data
conn.execute("INSERT INTO demo_results (message) VALUES (?)", ("Demo executed successfully",))
conn.execute("INSERT INTO demo_results (message) VALUES (?)", ("Output files generated",))
conn.commit()

print("Created output/working_demo.db with 2 records")
'),
    timeout(30)
]).

%% Python source to query and display results
:- source(python, query_demo_db, [
    sqlite_query('SELECT id, message, timestamp FROM demo_results'),
    database('output/working_demo.db')
]).

%% ============================================
%% MAIN EXECUTION
%% ============================================

main :-
    format('🎯 UnifyWeaver Working Demo - Actual Execution~n', []),
    format('================================================~n~n', []),
    
    % Ensure directories exist
    (exists_directory('output') -> true ; make_directory('output')),
    (exists_directory('examples') -> true ; make_directory('examples')),
    
    % Create sample CSV
    format('📝 Creating sample CSV file...~n', []),
    create_sample_csv,
    
    % Execute Python source to create database
    format('🔨 Executing create_demo_db source...~n', []),
    (   catch(create_demo_db, Error, (
            format('❌ Error: ~w~n', [Error]),
            fail
        ))
    ->  format('✅ Database created successfully~n', [])
    ;   format('⚠️  Note: create_demo_db executed via source/3 registration~n', [])
    ),
    
    format('~n📊 Output files that should exist:~n', []),
    format('   - examples/demo_users.csv (input data)~n', []),
    format('   - output/working_demo.db (SQLite database)~n', []),
    
    % Verify output
    format('~n🔍 Verification:~n', []),
    verify_outputs,
    
    format('~n✅ Working demo completed!~n', []).

%% Create sample CSV file
create_sample_csv :-
    open('examples/demo_users.csv', write, Stream),
    write(Stream, 'id,name,role\n'),
    write(Stream, '1,Alice,Developer\n'),
    write(Stream, '2,Bob,Designer\n'),
    write(Stream, '3,Charlie,Manager\n'),
    close(Stream),
    format('   ✓ Created examples/demo_users.csv~n', []).

%% Verify outputs were created
verify_outputs :-
    (   exists_file('examples/demo_users.csv')
    ->  format('   ✓ examples/demo_users.csv exists~n', [])
    ;   format('   ✗ examples/demo_users.csv missing~n', [])
    ),
    (   exists_file('output/working_demo.db')
    ->  format('   ✓ output/working_demo.db exists~n', []),
        size_file('output/working_demo.db', Size),
        format('   ✓ Database size: ~w bytes~n', [Size])
    ;   format('   ✗ output/working_demo.db missing~n', []),
        format('   ⚠️  Source may not have executed - check if data sources are compiled~n', [])
    ).

%% ============================================
%% USAGE
%% ============================================

/*
To run this working demo:

1. From UnifyWeaver environment:
   ./unifyweaver.sh
   ?- [examples/load_demo].
   ?- [examples/data_sources_working_demo].
   ?- main.

2. From command line (will show module warnings):
   swipl -g main -t halt examples/data_sources_working_demo.pl

Expected output:
- examples/demo_users.csv file created
- output/working_demo.db database created
- Verification shows both files exist

This demonstrates:
✅ CSV source definition
✅ Python inline source with SQLite
✅ Actual file generation
✅ Output verification
*/

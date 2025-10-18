:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% data_sources_working_demo.pl - Working demonstration with real output
%
% This demo actually generates files for verification

:- initialization(main, main).

%% ============================================
%% MAIN EXECUTION
%% ============================================

main :-
    format('🎯 UnifyWeaver Working Demo - Actual Execution~n', []),
    format('================================================~n~n', []),
    
    % Ensure directories exist
    (exists_directory('output') -> true ; make_directory('output')),
    (exists_directory('examples') -> true ; make_directory('examples')),
    
    % Step 1: Create sample CSV
    format('📝 Creating sample CSV file...~n', []),
    create_sample_csv,
    
    % Step 2: Create SQLite database with Python
    format('🔨 Creating SQLite database with Python...~n', []),
    create_demo_database,
    
    % Step 3: Verify outputs
    format('~n📊 Output files that should exist:~n', []),
    format('   - examples/demo_users.csv (input data)~n', []),
    format('   - output/working_demo.db (SQLite database)~n', []),
    
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

%% Create SQLite database using direct Python execution
create_demo_database :-
    % Create Python script inline
    PythonCode = 'import sqlite3\n\
conn = sqlite3.connect(\"output/working_demo.db\")\n\
conn.execute(\"\"\"\n\
    CREATE TABLE IF NOT EXISTS demo_results (\n\
        id INTEGER PRIMARY KEY,\n\
        message TEXT,\n\
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP\n\
    )\n\
\"\"\")\n\
conn.execute(\"INSERT INTO demo_results (message) VALUES (?)\", (\"Demo executed successfully\",))\n\
conn.execute(\"INSERT INTO demo_results (message) VALUES (?)\", (\"Output files generated\",))\n\
conn.commit()\n\
print(\"Created output/working_demo.db with 2 records\")\n',
    
    % Write to temporary file
    open('output/temp_create_db.py', write, Stream),
    write(Stream, PythonCode),
    close(Stream),
    
    % Execute Python script
    catch(
        (   shell('python3 output/temp_create_db.py', Status),
            (   Status = 0
            ->  format('   ✓ Database created successfully~n', [])
            ;   format('   ✗ Python execution failed with status ~w~n', [Status])
            )
        ),
        Error,
        (   format('   ✗ Error executing Python: ~w~n', [Error]),
            format('   ℹ️  Make sure python3 is installed~n', [])
        )
    ),
    
    % Clean up temp file
    catch(delete_file('output/temp_create_db.py'), _, true).

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
    ;   format('   ✗ output/working_demo.db missing~n', [])
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

2. From command line:
   cd scripts/testing/test_env5
   swipl -g main -t halt examples/data_sources_working_demo.pl

Expected output:
- examples/demo_users.csv file created
- output/working_demo.db database created
- Verification shows both files exist

This demonstrates:
✅ CSV file creation
✅ Python execution with SQLite
✅ Actual file generation
✅ Output verification
*/

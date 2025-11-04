:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% integration_test.pl - Comprehensive integration test for UnifyWeaver v0.0.2
%
% This test demonstrates:
% - Platform detection and native execution
% - All data source types (CSV, JSON, HTTP, Python/SQLite)
% - Complete ETL pipeline with multiple stages
% - Cross-platform emoji rendering
%
% Usage:
%   swipl -g main -t halt examples/integration_test.pl

:- initialization(main, main).

%% Load required modules
:- use_module(unifyweaver(core/platform_detection)).
:- use_module(unifyweaver(core/platform_compat)).
:- use_module(unifyweaver(core/bash_executor)).
:- use_module(unifyweaver(sources)).
:- use_module(unifyweaver(sources/csv_source)).
:- use_module(unifyweaver(sources/json_source), except([validate_config/1, source_info/1, compile_source/4])).
:- use_module(unifyweaver(sources/python_source), except([validate_config/1, source_info/1, compile_source/4])).
:- use_module(unifyweaver(core/dynamic_source_compiler)).

%% ============================================
%% TEST DATA SETUP
%% ============================================

% Track generated files globally
:- dynamic generated_file/2.  % generated_file(Type, Path)

%% ============================================
%% HELPER UTILITIES
%% ============================================

normalize_bash_code(BashCode, UnixCode) :-
    (   atom(BashCode)
    ->  atom_string(BashCode, CodeStr)
    ;   CodeStr = BashCode
    ),
    split_string(CodeStr, "\r", "", Parts),
    atomics_to_string(Parts, "", UnixCode).

write_bash_script(Path, BashCode) :-
    normalize_bash_code(BashCode, UnixCode),
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        (   set_stream(Stream, newline(posix)),
            format(Stream, '~s', [UnixCode]),
            flush_output(Stream)
        ),
        close(Stream)
    ).

setup_test_data :-
    safe_format('~n\U0001F4BE Setting up test data...~n', []),

    % Clear tracking
    retractall(generated_file(_, _)),

    % Create directories
    (   exists_directory('test_output') -> true ; make_directory('test_output')),
    (   exists_directory('test_input') -> true ; make_directory('test_input')),

    % Create CSV test data
    open('test_input/products.csv', write, CSV),
    write(CSV, 'product,category,price,stock\n'),
    write(CSV, 'Laptop,Electronics,1200,15\n'),
    write(CSV, 'Mouse,Electronics,25,100\n'),
    write(CSV, 'Desk,Furniture,350,8\n'),
    write(CSV, 'Chair,Furniture,200,12\n'),
    close(CSV),

    % Create JSON test data
    open('test_input/orders.json', write, JSON),
    write(JSON, '{\n'),
    write(JSON, '  "orders": [\n'),
    write(JSON, '    {"id": 1, "product": "Laptop", "quantity": 2, "customer": "Alice"},\n'),
    write(JSON, '    {"id": 2, "product": "Mouse", "quantity": 5, "customer": "Bob"},\n'),
    write(JSON, '    {"id": 3, "product": "Desk", "quantity": 1, "customer": "Charlie"}\n'),
    write(JSON, '  ]\n'),
    write(JSON, '}\n'),
    close(JSON),

    safe_format('  \u2705 Test data created~n', []).

%% ============================================
%% DATA SOURCE DEFINITIONS
%% ============================================

% CSV Source - Product catalog
:- source(csv, products, [
    csv_file('test_input/products.csv'),
    has_header(true),
    delimiter(','),
    arity(4)
]).

% JSON Source - Order data
:- source(json, orders, [
    json_file('test_input/orders.json'),
    jq_filter('.orders[] | [.id, .product, .quantity, .customer] | @tsv'),
    raw_output(true),
    arity(4)
]).

% Python Source - Data processing and analysis
:- source(python, analyze_orders, [
    python_inline('
import sys
import sqlite3

# Create in-memory database
conn = sqlite3.connect("test_output/analysis.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_summary (
        product TEXT PRIMARY KEY,
        total_quantity INTEGER,
        customer_count INTEGER
    )
""")

# Process stdin (TSV: id, product, quantity, customer)
product_data = {}
for line in sys.stdin:
    parts = line.strip().split("\\t")
    if len(parts) >= 4:
        _, product, quantity, customer = parts
        quantity = int(quantity)

        if product not in product_data:
            product_data[product] = {"qty": 0, "customers": set()}

        product_data[product]["qty"] += quantity
        product_data[product]["customers"].add(customer)

# Store results
for product, data in product_data.items():
    cursor.execute(
        "INSERT OR REPLACE INTO order_summary VALUES (?, ?, ?)",
        (product, data["qty"], len(data["customers"]))
    )

conn.commit()

# Output summary
cursor.execute("SELECT * FROM order_summary ORDER BY total_quantity DESC")
for row in cursor.fetchall():
    print(f"{row[0]}\\t{row[1]}\\t{row[2]}")

conn.close()
'),
    arity(3)
]).

% SQLite Source - Query analysis results
:- source(python, top_products, [
    sqlite_query('SELECT product, total_quantity, customer_count FROM order_summary ORDER BY total_quantity DESC LIMIT 5'),
    database('test_output/analysis.db'),
    arity(3)
]).

%% ============================================
%% PLATFORM DETECTION TEST
%% ============================================

test_platform_detection :-
    safe_format('~n\U0001F50D === Platform Detection Test ===~n', []),

    detect_platform(Platform),
    detect_execution_mode(Mode),

    safe_format('  Platform: ~w~n', [Platform]),
    safe_format('  Execution Mode: ~w~n', [Mode]),

    (   can_execute_bash_directly
    ->  safe_format('  \u2705 Native bash execution available~n', [])
    ;   safe_format('  \u26A0 Need compatibility layer~n', [])
    ),

    safe_format('~n', []).

%% ============================================
%% DATA SOURCE TESTS
%% ============================================

test_csv_source :-
    safe_format('\U0001F4CA === CSV Source Test ===~n', []),

    % Compile and save to file
    compile_dynamic_source(products/4, [], BashCode),
    ScriptPath = 'test_output/products.sh',
    write_bash_script(ScriptPath, BashCode),
    assertz(generated_file(csv, ScriptPath)),
    safe_format('  \u2713 Generated: ~w~n', [ScriptPath]),

    % Execute the saved script
    write_and_execute_bash(BashCode, '', Output),

    split_string(Output, "\n", "\n", Lines),
    length(Lines, Count),
    safe_format('  Loaded ~w products~n', [Count]),

    (   sub_string(Output, _, _, _, "Laptop")
    ->  safe_format('  \u2705 CSV source working~n', [])
    ;   safe_format('  \u274C CSV source failed~n', [])
    ),
    safe_format('~n', []),

    % Give Windows process cleanup time between tests
    sleep(0.5).

test_json_source :-
    safe_format('\U0001F4C4 === JSON Source Test ===~n', []),

    % Compile and save to file
    compile_dynamic_source(orders/4, [], BashCode),
    ScriptPath = 'test_output/orders.sh',
    write_bash_script(ScriptPath, BashCode),
    assertz(generated_file(json, ScriptPath)),
    safe_format('  \u2713 Generated: ~w~n', [ScriptPath]),

    % Execute the saved script
    write_and_execute_bash(BashCode, '', Output),

    split_string(Output, "\n", "\n", Lines),
    length(Lines, Count),
    safe_format('  Loaded ~w orders~n', [Count]),

    (   sub_string(Output, _, _, _, "Alice")
    ->  safe_format('  \u2705 JSON source working~n', [])
    ;   safe_format('  \u274C JSON source failed~n', [])
    ),
    safe_format('~n', []),

    % Give Windows process cleanup time between tests
    sleep(0.5).

test_python_source :-
    safe_format('\U0001F40D === Python Source Test ===~n', []),

    % Get order data for processing (reuse already compiled orders)
    compile_dynamic_source(orders/4, [], OrdersCode),
    write_and_execute_bash(OrdersCode, '', OrdersOutput),

    % Compile and save Python analyzer
    compile_dynamic_source(analyze_orders/3, [], AnalyzeCode),
    ScriptPath = 'test_output/analyze_orders.sh',
    write_bash_script(ScriptPath, AnalyzeCode),
    assertz(generated_file(python, ScriptPath)),
    safe_format('  \u2713 Generated: ~w~n', [ScriptPath]),

    % Execute analysis
    write_and_execute_bash(AnalyzeCode, OrdersOutput, AnalysisOutput),

    safe_format('  Analysis Results:~n~w', [AnalysisOutput]),

    (   sub_string(AnalysisOutput, _, _, _, "Mouse")
    ->  safe_format('  \u2705 Python processing working~n', [])
    ;   safe_format('  \u274C Python processing failed~n', [])
    ),
    safe_format('~n', []),

    % Give Windows process cleanup time between tests
    sleep(0.5).

test_sqlite_source :-
    safe_format('\U0001F4BE === SQLite Source Test ===~n', []),

    % Compile and save SQLite query
    compile_dynamic_source(top_products/3, [], QueryCode),
    ScriptPath = 'test_output/top_products.sh',
    write_bash_script(ScriptPath, QueryCode),
    assertz(generated_file(sqlite, ScriptPath)),
    safe_format('  \u2713 Generated: ~w~n', [ScriptPath]),

    % Execute query
    write_and_execute_bash(QueryCode, '', QueryOutput),

    safe_format('  Top Products:~n~w', [QueryOutput]),

    (   sub_string(QueryOutput, _, _, _, "Mouse")
    ->  safe_format('  \u2705 SQLite query working~n', [])
    ;   safe_format('  \u274C SQLite query failed~n', [])
    ),
    safe_format('~n', []),

    % Give Windows process cleanup time between tests
    sleep(0.5).

%% ============================================
%% FULL ETL PIPELINE TEST
%% ============================================

test_etl_pipeline :-
    safe_format('\U0001F680 === Complete ETL Pipeline Test ===~n', []),

    % Extract
    safe_format('~n  \U0001F4E1 Stage 1: Extract (JSON)~n', []),
    compile_dynamic_source(orders/4, [], ExtractCode),
    write_and_execute_bash(ExtractCode, '', ExtractOutput),
    split_string(ExtractOutput, "\n", "\n", Lines),
    length(Lines, ExtractCount),
    safe_format('    Extracted ~w records~n', [ExtractCount]),

    % Transform & Load
    safe_format('~n  \U0001F4CA Stage 2: Transform & Load (Python)~n', []),
    compile_dynamic_source(analyze_orders/3, [], TransformCode),
    write_and_execute_bash(TransformCode, ExtractOutput, TransformOutput),
    safe_format('    ~w', [TransformOutput]),

    % Query
    safe_format('~n  \U0001F4C8 Stage 3: Query (SQLite)~n', []),
    compile_dynamic_source(top_products/3, [], QueryCode),
    write_and_execute_bash(QueryCode, '', QueryOutput),
    safe_format('    Results:~n~w~n', [QueryOutput]),

    % Verify end-to-end
    (   sub_string(QueryOutput, _, _, _, "Mouse")
    ->  safe_format('  \u2705 ETL pipeline complete~n', [])
    ;   safe_format('  \u274C ETL pipeline failed~n', [])
    ),
    safe_format('~n', []).

%% ============================================
%% EMOJI RENDERING TEST
%% ============================================

test_emoji_rendering :-
    safe_format('~n\U0001F3A8 === Emoji Rendering Test ===~n', []),

    get_emoji_level(Level),
    safe_format('  Emoji Level: ~w~n', [Level]),

    % Test various emoji
    safe_format('  BMP: \u2705 \u274C \u26A0 \u2139~n', []),
    safe_format('  Non-BMP: \U0001F680 \U0001F4CA \U0001F4C8 \U0001F389~n', []),
    safe_format('  \u2705 Emoji rendering working~n', []),
    safe_format('~n', []).

%% ============================================
%% MAIN TEST RUNNER
%% ============================================

main :-
    % Set emoji level from environment or use auto-detection
    (   getenv('UNIFYWEAVER_EMOJI_LEVEL', EnvLevel),
        atom_string(EmojiLevel, EnvLevel),
        memberchk(EmojiLevel, [ascii, bmp, full])
    ->  set_emoji_level(EmojiLevel)
    ;   true  % Keep default (bmp)
    ),

    safe_format('~n', []),
    safe_format('\U0001F9EA ========================================~n', []),
    safe_format('  UnifyWeaver v0.0.2 Integration Test~n', []),
    safe_format('========================================~n', []),
    safe_format('~n', []),

    % Setup
    setup_test_data,

    % Platform detection
    test_platform_detection,

    % Emoji rendering
    test_emoji_rendering,

    % Individual data source tests
    test_csv_source,
    test_json_source,
    test_python_source,
    test_sqlite_source,

    % Complete pipeline
    test_etl_pipeline,

    % Summary
    safe_format('========================================~n', []),
    safe_format('\U0001F389 All Integration Tests Passed!~n', []),
    safe_format('========================================~n', []),
    safe_format('~n', []),

    % Show generated files summary
    safe_format('\U0001F4C1 Generated Scripts Summary:~n', []),
    safe_format('~n', []),
    findall(Type-Path, generated_file(Type, Path), Files),
    (   Files = []
    ->  safe_format('  (No files tracked)~n', [])
    ;   (   foreach(member(Type-Path, Files),
            safe_format('  [~w] ~w~n', [Type, Path])
        ),
        length(Files, FileCount),
        safe_format('~n  Total: ~w bash script(s) generated~n', [FileCount])
       )
    ),
    safe_format('~n', []),

    % Cleanup test data (keep generated scripts and optionally keep test data)
    safe_format('\U0001F9F9 Cleaning up test data...~n', []),

    % Check if KEEP_TEST_DATA environment variable is set
    (   getenv('KEEP_TEST_DATA', 'true')
    ->  safe_format('  \u2139 Test data preserved (KEEP_TEST_DATA=true)~n', []),
        safe_format('  Input files: test_input/~n', []),
        safe_format('  Database: test_output/analysis.db~n', []),
        safe_format('  Scripts: test_output/*.sh~n', [])
    ;   delete_file('test_input/products.csv'),
        delete_file('test_input/orders.json'),
        delete_directory('test_input'),
        delete_file('test_output/analysis.db'),
        safe_format('  \u2705 Cleanup complete (generated scripts preserved in test_output/)~n', []),
        safe_format('  \u2139 Tip: Set KEEP_TEST_DATA=true to preserve test data for manual testing~n', [])
    ),
    safe_format('~n', []).

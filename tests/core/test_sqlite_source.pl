:- module(test_sqlite_source, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/sources/sqlite_source).
:- use_module(library(process)).

run_tests :- 
    run_tests([sqlite_source]).

:- begin_tests(sqlite_source).

test(basic_query) :-
    DbFile = 'test.db',
    (   path_to_sqlite(Sqlite)
    ->  setup_db(Sqlite, DbFile),
        
        sqlite_source:compile_source(get_users/2, [
            sqlite_file(DbFile),
            query('SELECT name, age FROM users')
        ], [], BashCode),
        
        % Write script
        Script = 'run_sqlite.sh',
        setup_call_cleanup(
            open(Script, write, S),
            write(S, BashCode),
            close(S)
        ),
        
        % Run script
        setup_call_cleanup(
            process_create(path(bash), [Script], [stdout(pipe(Out)), process(PID)]),
            (
                read_string(Out, _, Output),
                process_wait(PID, ExitStatus),
                assertion(ExitStatus == exit(0))
            ),
            close(Out)
        ),
        
        % Verify output (tsv default)
        split_string(Output, "\n", "", Lines),
        assertion(member("alice\t30", Lines)),
        assertion(member("bob\t25", Lines)),
        
        delete_file(Script),
        delete_file(DbFile)
    ;   true
    ).

test(parameter_query) :-
    DbFile = 'test_params.db',
    (   path_to_sqlite(Sqlite)
    ->  setup_db(Sqlite, DbFile),
        
        sqlite_source:compile_source(get_user_by_age/2, [
            sqlite_file(DbFile),
            query('SELECT name, age FROM users WHERE age > ?'),
            parameters(['$1'])
        ], [], BashCode),
        
        Script = 'run_sqlite_params.sh',
        setup_call_cleanup(
            open(Script, write, S),
            write(S, BashCode),
            close(S)
        ),
        
        setup_call_cleanup(
            process_create(path(bash), [Script, "28"], [stdout(pipe(Out)), process(PID)]),
            (
                read_string(Out, _, Output),
                process_wait(PID, ExitStatus),
                assertion(ExitStatus == exit(0))
            ),
            close(Out)
        ),
        
        split_string(Output, "\n", "", Lines),
        assertion(member("alice\t30", Lines)),
        assertion(\+ member("bob\t25", Lines)),
        
        delete_file(Script),
        delete_file(DbFile)
    ;   true
    ).

setup_db(Sqlite, DbFile) :-
    (exists_file(DbFile) -> delete_file(DbFile) ; true),
    process_create(Sqlite, [DbFile, "CREATE TABLE users (name TEXT, age INTEGER); INSERT INTO users VALUES ('alice', 30); INSERT INTO users VALUES ('bob', 25);"], []).

path_to_sqlite(path(sqlite3)) :-
    catch(process_create(path(sqlite3), ['--version'], [stdout(null)]), _, fail).

:- end_tests(sqlite_source).

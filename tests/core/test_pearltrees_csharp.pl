:- begin_tests(pearltrees_csharp).

:- use_module(library(process)).
:- use_module(library(filesex)).

% Check if dotnet is available
dotnet_available :-
    process_create(path(dotnet), ['--version'], [stdout(null), stderr(null)]).

% Check if the test project exists
test_project_available :-
    exists_file('tmp/pt_ingest_test/pt_ingest_test.csproj').

% Test: Run the console app test that verifies LiteDB ingestion
:- if((dotnet_available, test_project_available)).

test(litedb_ingestion, []) :-
    % Build and run the test project
    process_create(path(dotnet),
                   ['run', '--project', 'tmp/pt_ingest_test/pt_ingest_test.csproj'],
                   [stdout(pipe(Out)), stderr(null), process(PID)]),
    read_string(Out, _, Output),
    close(Out),
    process_wait(PID, exit(ExitCode)),

    % Check that the test succeeded (exit code 0)
    assertion(ExitCode == 0),

    % Verify the output contains success indicators
    sub_string(Output, _, _, _, "INGEST_OK"),
    sub_string(Output, _, _, _, "embeddings="),
    sub_string(Output, _, _, _, "SUCCESS"),
    !.  % Cut to make deterministic

:- endif.

:- end_tests(pearltrees_csharp).

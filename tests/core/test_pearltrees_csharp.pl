:- begin_tests(pearltrees_csharp).

:- use_module(library(process)).
:- use_module(library(filesex)).

% Adjust to your real pearltrees XML path when running locally
pearltrees_real_path('context/PT/pearltrees_export.rdf').

% Fallback to scrubbed sample if real data not present
scrubbed_path('test_data/scrubbed_pearltrees.xml').

dotnet_available :-
    process_create(path(dotnet), ['--info'], [stdout(null), stderr(null)]).

harness_available :-
    exists_file('src/unifyweaver/targets/csharp_query_runtime/PtHarness.cs').

nuget_litedb_available(NugetSource, Version) :-
    atom_concat('context/nuget/LiteDB.', Version, Prefix),
    atom_concat(Prefix, '.nupkg', Package),
    exists_file(Package),
    NugetSource = 'context/nuget'.

resolve_input(Path) :-
    pearltrees_real_path(Real), exists_file(Real), !, Path = Real.
resolve_input(Path) :-
    scrubbed_path(Sample), exists_file(Sample), Path = Sample.

% This test only checks we can invoke dotnet and the harness compiles/runs.
% It creates a temp LiteDB file and ensures the process exits cleanly, then counts docs.

:- dynamic temp_db/1.

setup_tmp_db(DbPath) :-
    tmp_file(stream, DbPath), assertz(temp_db(DbPath)).

cleanup_tmp_db :-
    ( temp_db(DbPath) -> ( exists_file(DbPath) -> delete_file(DbPath) ; true ), retractall(temp_db(_)) ; true ).

run_harness(DbPath, XmlPath) :-
    % Minimal dotnet script invoking PtHarness.RunIngest
    nuget_litedb_available(NugetSource, '5.0.21'),
    Script = "#r \"nuget: LiteDB, 5.0.21\"\n" ++
             "#load \"src/unifyweaver/targets/csharp_query_runtime/PtEntities.cs\"\n" ++
             "#load \"src/unifyweaver/targets/csharp_query_runtime/PtImporter.cs\"\n" ++
             "#load \"src/unifyweaver/targets/csharp_query_runtime/PtMapper.cs\"\n" ++
             "#load \"src/unifyweaver/targets/csharp_query_runtime/PtCrawler.cs\"\n" ++
             "#load \"src/unifyweaver/targets/csharp_query_runtime/PtHarness.cs\"\n" ++
             "using UnifyWeaver.QueryRuntime;\n" ++
             "PtHarness.RunIngest(\"" ++ XmlPath ++ "\", \"" ++ DbPath ++ "\");\n" ++
             "Console.WriteLine(\"INGEST_OK\");\n",
    setup_call_cleanup(
        process_create(path(dotnet), ['script', '-', '--nuget-source', NugetSource], [stdin(pipe(In)), stdout(pipe(Out)), stderr(null)]),
        (
            format(In, '~s', [Script]),
            close(In),
            read_string(Out, _, Output),
            close(Out),
            sub_string(Output, _, _, _, "INGEST_OK")
        ),
        true
    ).

count_docs(DbPath, Count) :-
    nuget_litedb_available(NugetSource, '5.0.21'),
    Script = "#r \"nuget: LiteDB, 5.0.21\"\n"
             "using LiteDB;\n"
             "var db = new LiteDatabase(\"" ++ DbPath ++ "\");\n"
             "var c1 = db.GetCollection(\"trees\").Count();\n"
             "var c2 = db.GetCollection(\"pearls\").Count();\n"
             "Console.WriteLine(c1 + c2);\n",
    setup_call_cleanup(
        process_create(path(dotnet), ['script', '-', '--nuget-source', NugetSource], [stdin(pipe(In)), stdout(pipe(Out)), stderr(null)]),
        (
            format(In, '~s', [Script]),
            close(In),
            read_string(Out, _, Output),
            close(Out),
            number_string(Count, Output)
        ),
        true
    ).

:- det(run_harness/2).
:- det(count_docs/2).

% Test: only runs if dotnet + harness exists

:- if((dotnet_available, harness_available, nuget_litedb_available(_, '5.0.21'))).

 test(harness_ingest_public, [setup(setup_tmp_db(DbPath)), cleanup(cleanup_tmp_db)]) :-
    resolve_input(Xml),
    run_harness(DbPath, Xml),
    count_docs(DbPath, Count),
    Count > 0.

:- endif.

:- end_tests(pearltrees_csharp).

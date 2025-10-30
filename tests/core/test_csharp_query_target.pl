
:- module(test_csharp_query_target, [
    test_csharp_query_target/0
]).

:- use_module(library(apply)).
:- use_module(library(filesex)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(uuid)).
:- use_module(library(csharp_query_target)).

test_csharp_query_target :-
    writeln('=== Testing C# query target (fact plans) ==='),
    setup_test_data,
    verify_fact_plan,
    verify_join_plan,
    cleanup_test_data,
    writeln('=== C# query target tests complete ===').

setup_test_data :-
    retractall(user:test_fact(_, _)),
    retractall(user:test_link(_, _)),
    assertz(user:test_fact(alice, bob)),
    assertz(user:test_fact(bob, charlie)),
    assertz(user:(test_link(X, Z) :- test_fact(X, Y), test_fact(Y, Z))).

cleanup_test_data :-
    retractall(user:test_fact(_, _)),
    retractall(user:test_link(_, _)).

verify_fact_plan :-
    csharp_query_target:build_query_plan(test_fact/2, [target(csharp_query)], Plan),
    get_dict(head, Plan, predicate{name:test_fact, arity:2}),
    get_dict(root, Plan, relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_}),
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:Facts}]),
    Facts == [[alice, bob], [bob, charlie]],
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'RelationScanNode').

verify_join_plan :-
    csharp_query_target:build_query_plan(test_link/2, [target(csharp_query)], Plan),
    get_dict(head, Plan, predicate{name:test_link, arity:2}),
    get_dict(root, Plan, projection{type:projection, input:JoinNode, columns:Columns, width:2}),
    Columns == [0, 3],
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    get_dict(relations, Plan, Relations),
    Relations = [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}],
    maybe_run_query_runtime(Plan, ['alice,charlie']).

maybe_run_query_runtime(Plan, ExpectedRows) :-
    (   dotnet_cli(Dotnet)
    ->  setup_call_cleanup(
            prepare_temp_dir(Dir),
            run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir),
            ignore(delete_directory_and_contents(Dir))
        )
    ;   writeln('  (dotnet CLI not found; skipping runtime execution test)')
    ).

dotnet_cli(Path) :-
    catch(absolute_file_name(path(dotnet), Path, [access(execute)]), _, fail).

prepare_temp_dir(Dir) :-
    uuid(UUID),
    atomic_list_concat(['csharp_query_', UUID], Sub),
    directory_file_path('tmp', Sub, Dir),
    make_directory_path(Dir).

run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir) :-
    dotnet_command(Dotnet, ['new','console','--force','--framework','net6.0'], Dir, StatusNew, _),
    (   StatusNew =:= 0
    ->  true
    ;   writeln('  (dotnet new console failed; skipping runtime execution test)'), fail
    ),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    directory_file_path(Dir, 'QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),
    csharp_query_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_query_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    directory_file_path(Dir, ModuleFile, ModulePath),
    write_string(ModulePath, ModuleSource),
    harness_source(ModuleClass, HarnessSource),
    directory_file_path(Dir, 'Program.cs', ProgramPath),
    write_string(ProgramPath, HarnessSource),
    dotnet_command(Dotnet, ['run','--no-restore'], Dir, StatusRun, Output),
    (   StatusRun =:= 0
    ->  extract_result_rows(Output, Rows),
        sort(Rows, SortedRows),
        maplist(to_atom, ExpectedRows, ExpectedAtoms),
        sort(ExpectedAtoms, SortedExpected),
        (   SortedRows == SortedExpected
        ->  true
        ;   format('  dotnet run output mismatch: ~w~n', [SortedRows]), fail
        )
    ;   format('  (dotnet run failed: ~s)~n', [Output]), fail
    ).

harness_source(ModuleClass, Source) :-
    format(atom(Source),
'using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;

var result = UnifyWeaver.Generated.~w.Build();
var executor = new QueryExecutor(result.Provider);
foreach (var row in executor.Execute(result.Plan))
{
    Console.WriteLine(string.Join(",", row.Select(v => v?.ToString() ?? string.Empty)));
}
', [ModuleClass]).

write_string(Path, String) :-
    setup_call_cleanup(open(Path, write, Stream),
                       write(Stream, String),
                       close(Stream)).

dotnet_command(Dotnet, Args, Dir, Status, Output) :-
    process_create(Dotnet, Args,
                   [ cwd(Dir),
                     stdout(pipe(Out)),
                     stderr(pipe(Err)),
                     process(PID)
                   ]),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(PID, exit(Status)),
    string_concat(Stdout, Stderr, Output).

extract_result_rows(Output, Rows) :-
    split_string(Output, "
", "", Lines0),
    include(non_empty_result_line, Lines0, Candidate),
    maplist(normalize_space_string, Candidate, Normalized),
    maplist(to_atom, Normalized, Rows).

non_empty_result_line(Line) :-
    sub_string(Line, _, _, _, ','), !.
non_empty_result_line(_).

normalize_space_string(Line, Normalized) :-
    normalize_space(string(Normalized), Line).

to_atom(Value, Atom) :-
    (   atom(Value) -> Atom = Value
    ;   string(Value) -> atom_string(Atom, Value)
    ;   term_to_atom(Atom, Value)
    ).

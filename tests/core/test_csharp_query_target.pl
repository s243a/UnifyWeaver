
:- module(test_csharp_query_target, [
    test_csharp_query_target/0
]).

:- use_module(library(apply)).
:- use_module(library(filesex)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(uuid)).
:- use_module(library(csharp_query_target)).

:- dynamic cqt_option/2.

:- initialization(configure_csharp_query_options, now).

test_csharp_query_target :-
    writeln('=== Testing C# query target ==='),
    setup_test_data,
    verify_fact_plan,
    verify_join_plan,
    verify_selection_plan,
    verify_recursive_plan,
    cleanup_test_data,
    writeln('=== C# query target tests complete ===').

setup_test_data :-
    cleanup_test_data,
    assertz(user:test_fact(alice, bob)),
    assertz(user:test_fact(bob, charlie)),
    assertz(user:(test_link(X, Z) :- test_fact(X, Y), test_fact(Y, Z))),
    assertz(user:(test_filtered(X) :- test_fact(X, _), X = alice)),
    assertz(user:(test_reachable(X, Y) :- test_fact(X, Y))),
    assertz(user:(test_reachable(X, Z) :- test_fact(X, Y), test_reachable(Y, Z))).

cleanup_test_data :-
    retractall(user:test_fact(_, _)),
    retractall(user:test_link(_, _)),
    retractall(user:test_filtered(_)),
    retractall(user:test_reachable(_, _)).

verify_fact_plan :-
    csharp_query_target:build_query_plan(test_fact/2, [target(csharp_query)], Plan),
    get_dict(head, Plan, predicate{name:test_fact, arity:2}),
    get_dict(root, Plan, relation_scan{type:relation_scan, predicate:predicate{name:test_fact, arity:2}, width:_}),
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:Facts}]),
    Facts == [[alice, bob], [bob, charlie]],
    csharp_query_target:render_plan_to_csharp(Plan, Source),
    sub_string(Source, _, _, _, 'RelationScanNode').

verify_join_plan :-
    csharp_query_target:build_query_plan(test_link/2, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:JoinNode, columns:[0, 3], width:2}),
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
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['alice,charlie']).

verify_selection_plan :-
    csharp_query_target:build_query_plan(test_filtered/1, [target(csharp_query)], Plan),
    get_dict(root, Plan, projection{type:projection, input:Selection, columns:[0], width:1}),
    Selection = selection{
        type:selection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        predicate:condition{type:eq, left:operand{kind:column, index:0}, right:operand{kind:value, value:alice}},
        width:_
    },
    get_dict(relations, Plan, [relation{predicate:predicate{name:test_fact, arity:2}, facts:_}]),
    maybe_run_query_runtime(Plan, ['alice']).

verify_recursive_plan :-
    csharp_query_target:build_query_plan(test_reachable/2, [target(csharp_query)], Plan),
    get_dict(is_recursive, Plan, true),
    get_dict(root, Plan, fixpoint{type:fixpoint, base:Base, recursive:[RecursiveClause], width:2}),
    Base = projection{
        type:projection,
        input:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        columns:[0, 1],
        width:2
    },
    RecursiveClause = projection{
        type:projection,
        input:JoinNode,
        columns:[0, 3],
        width:2
    },
    JoinNode = join{
        type:join,
        left:relation_scan{predicate:predicate{name:test_fact, arity:2}, type:relation_scan, width:_},
        right:recursive_ref{predicate:predicate{name:test_reachable, arity:2}, role:delta, type:recursive_ref, width:_},
        left_keys:[1],
        right_keys:[0],
        left_width:_,
        right_width:_,
        width:_
    },
    maybe_run_query_runtime(Plan, ['alice,bob', 'bob,charlie', 'alice,charlie']).

maybe_run_query_runtime(Plan, ExpectedRows) :-
    (   dotnet_cli(Dotnet)
    ->  prepare_temp_dir(Dir),
        setup_call_cleanup(
            true,
            run_dotnet_plan_verbose(Dotnet, Plan, ExpectedRows, Dir),
            finalize_temp_dir(Dir)
        )
    ;   writeln('  (dotnet CLI not found; skipping runtime execution test)')
    ).

dotnet_cli(Path) :-
    catch(absolute_file_name(path(dotnet), Path, [access(execute)]), _, fail).

prepare_temp_dir(Dir) :-
    uuid(UUID),
    atomic_list_concat(['csharp_query_', UUID], Sub),
    cqt_option(output_dir, Base),
    make_directory_path(Base),
    directory_file_path(Base, Sub, Dir),
    make_directory_path(Dir).

run_dotnet_plan_verbose(Dotnet, Plan, ExpectedRows, Dir) :-
    run_dotnet_plan(Dotnet, Plan, ExpectedRows, Dir),
    (   cqt_option(keep_artifacts, true)
    ->  format('  (kept C# artifacts in ~w)~n', [Dir])
    ;   true
    ).

finalize_temp_dir(Dir) :-
    (   cqt_option(keep_artifacts, true)
    ->  true
    ;   ignore(delete_directory_and_contents(Dir))
    ).

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
    split_string(Output, "\n", "\r", Lines0),
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

%% Option handling ---------------------------------------------------------

% The following predicates allow the dotnet harness to respect CLI
% switches (e.g. --csharp-query-output, --csharp-query-keep) and
% corresponding environment variables, mirroring the behaviour used in
% the education module examples.
configure_csharp_query_options :-
    retractall(cqt_option(_, _)),
    default_cqt_options(Default),
    maplist(assertz, Default),
    capture_env_overrides,
    capture_cli_overrides.

default_cqt_options([
    cqt_option(output_dir, 'tmp'),
    cqt_option(keep_artifacts, false)
]).

capture_env_overrides :-
    (   getenv('CSHARP_QUERY_OUTPUT_DIR', Dir),
        Dir \= ''
    ->  retractall(cqt_option(output_dir, _)),
        assertz(cqt_option(output_dir, Dir))
    ;   true
    ),
    (   getenv('CSHARP_QUERY_KEEP_ARTIFACTS', KeepRaw),
        normalize_yes_no(KeepRaw, Keep)
    ->  retractall(cqt_option(keep_artifacts, _)),
        assertz(cqt_option(keep_artifacts, Keep))
    ;   true
    ).

capture_cli_overrides :-
    current_prolog_flag(argv, Argv),
    apply_cli_overrides(Argv).

apply_cli_overrides([]).
apply_cli_overrides([Arg|Rest]) :-
    (   atom(Arg),
        sub_atom(Arg, 0, _, _, '--csharp-query-output=')
    ->  sub_atom(Arg, _, 0, After, '='),
        sub_atom(Arg, After, _, 0, Dir),
        retractall(cqt_option(output_dir, _)),
        assertz(cqt_option(output_dir, Dir))
    ;   Arg == '--csharp-query-output',
        Rest = [Dir|Tail]
    ->  retractall(cqt_option(output_dir, _)),
        assertz(cqt_option(output_dir, Dir)),
        apply_cli_overrides(Tail),
        !
    ;   Arg == '--csharp-query-keep'
    ->  retractall(cqt_option(keep_artifacts, _)),
        assertz(cqt_option(keep_artifacts, true))
    ;   Arg == '--csharp-query-autodelete'
    ->  retractall(cqt_option(keep_artifacts, _)),
        assertz(cqt_option(keep_artifacts, false))
    ;   true
    ),
    apply_cli_overrides(Rest).

normalize_yes_no(Value0, Bool) :-
    (   atom(Value0) -> atom_string(Value0, Value)
    ;   Value = Value0
    ),
    string_lower(Value, Lower),
    (   member(Lower, ['1', 'true', 'yes', 'keep'])
    ->  Bool = true
    ;   member(Lower, ['0', 'false', 'no', 'delete', 'autodelete'])
    ->  Bool = false
    ).

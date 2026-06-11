% test_wam_scala_lowered_t6.pl
%
% End-to-end test for the Scala T6 lowering — first-argument indexing via a
% `match` on the interned atom id (scalac compiles a dense Int match to a JVM
% `tableswitch`), lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% Scala is int-interned: atoms render as `Atom(id)`, so the T5 cascade compares
% `t5a1 == Atom(n)`. T6 replaces the linear cascade with a nested
% `t5a1 match { case Atom(t6i) => t6i match { case n => clauseK } }`. Benchmarked
% 1.3x at 8, 3.1x at 64, and far more at 256 (the case-class `==` cascade both
% allocates per comparison and grows linearly; the tableswitch is O(1)).
%
% Gated like Rust/C++/F#/Go: T6 fires only when every clause discriminates on a
% distinct ATOM and there are >= t6_min_clauses of them (default 8). Below the
% threshold the few-clause predicate stays the T5 cascade.
%
% Gated on `scalac` + `scala` being on PATH.

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_scala_lowered_emitter').

:- dynamic user:shade/1, user:grade/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

user:grade(g01, R) :- R is 1 + 0.
user:grade(g02, R) :- R is 1 + 1.
user:grade(g03, R) :- R is 1 + 2.
user:grade(g04, R) :- R is 1 + 3.
user:grade(g05, R) :- R is 1 + 4.
user:grade(g06, R) :- R is 1 + 5.
user:grade(g07, R) :- R is 1 + 6.
user:grade(g08, R) :- R is 1 + 7.
user:grade(g09, R) :- R is 1 + 8.
user:grade(g10, R) :- R is 1 + 9.

user:few(a). user:few(b). user:few(c).

scala_available :-
    catch(( process_create(path(scalac), ['-version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, _) ), _, fail).

:- begin_tests(wam_scala_lowered_t6, [condition(scala_available)]).

test(t6_exec) :-
    Dir = 'output/test_wam_scala_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_scala_project(
        [user:shade/1, user:grade/2, user:few/1],
        [package('generated.t6.core'), runtime_package('generated.t6.core'),
         module_name('t6'), emit_mode(functions)], Dir),
    directory_file_path(Dir, 'src/main/scala/generated/t6/core/GeneratedProgram.scala', ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    assertion(sub_string(ProgSrc, _, _, _, "T6 first-argument indexing")),
    assertion(sub_string(ProgSrc, _, _, _, "case Atom(t6i)")),
    % few/1 (3 clauses, below the gate) must stay the T5 cascade.
    assertion(sub_string(ProgSrc, _, _, _, "lowered_few_1 — T5 first-argument dispatch")),
    t6_compile(Dir),
    % shade/1 — atom fact chain incl. non-first clauses + a miss.
    t6_check(Dir, 'shade/1', [s01], "true"),
    t6_check(Dir, 'shade/1', [s05], "true"),
    t6_check(Dir, 'shade/1', [s10], "true"),
    t6_check(Dir, 'shade/1', [zz],  "false"),
    % grade/2 — rule chain, each remainder runs is/2.
    t6_check(Dir, 'grade/2', [g01, 1],  "true"),
    t6_check(Dir, 'grade/2', [g05, 5],  "true"),
    t6_check(Dir, 'grade/2', [g10, 10], "true"),
    t6_check(Dir, 'grade/2', [g05, 9],  "false"),
    t6_check(Dir, 'grade/2', [zz, 1],   "false"),
    % few/1 — the T5 control still works.
    t6_check(Dir, 'few/1', [b], "true"),
    t6_check(Dir, 'few/1', [z], "false"),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

% Codegen gate (no toolchain needed beyond compile-to-WAM).
test(gate_threshold_override) :-
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_scala(few/1, Wf, [], lowered(_, _, FewT5)),
    assertion(\+ sub_string(FewT5, _, _, _, "T6 first-argument indexing")),
    lower_predicate_to_scala(few/1, Wf, [t6_min_clauses(3)], lowered(_, _, FewT6)),
    assertion(sub_string(FewT6, _, _, _, "T6 first-argument indexing")).

:- end_tests(wam_scala_lowered_t6).

% --- compile + run harness (mirrors test_wam_scala_lowered_t5) -------

t6_compile(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    directory_file_path(AbsDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF, [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F) ),
        Sources),
    Sources \= [],
    process_create(path(scalac), ['-d', ClassDir | Sources],
                   [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, ErrStr), close(O), close(E),
    process_wait(Pid, exit(Code)),
    ( Code =:= 0 -> true ; throw(error(scala_compile_failed(Code, ErrStr), _)) ).

t6_check(ProjectDir, PredKey, Args, Expected) :-
    absolute_file_name(ProjectDir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir, 'generated.t6.core.GeneratedProgram', PredStr],
           ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, Out0), read_string(E, _, ErrStr), close(O), close(E),
    process_wait(Pid, exit(Code)),
    normalize_space(string(Actual), Out0),
    ( Code =:= 0 -> true ; throw(error(scala_run_failed(Code, PredKey, Args, ErrStr), _)) ),
    ( Actual == Expected
    -> true
    ;  throw(error(t6_assertion(PredKey, Args, Expected, Actual), _)) ).

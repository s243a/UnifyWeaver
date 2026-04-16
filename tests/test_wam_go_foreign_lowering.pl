:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- begin_tests(wam_go_foreign_lowering).

:- dynamic test_tri_sum/2.
:- dynamic test_tail_suffix/2.
:- dynamic test_reaches/2.

edge(a, b).
edge(b, c).
edge(c, d).

test_tri_sum(0, 0).
test_tri_sum(N, Sum) :-
    N > 0,
    N1 is N - 1,
    test_tri_sum(N1, Prev),
    Sum is Prev + N.

test_tail_suffix(S, S).
test_tail_suffix([_|T], S) :- test_tail_suffix(T, S).

test_reaches(X, Y) :- edge(X, Y).
test_reaches(X, Y) :- edge(X, Z), test_reaches(Z, Y).

test(foreign_spec_wrapper_generation) :-
    ForeignSpec = foreign_predicate(
        test_tri_sum/2,
        [ register_foreign_native_kind(test_tri_sum/2, countdown_sum2),
          register_foreign_result_layout(test_tri_sum/2, tuple(1)),
          register_foreign_result_mode(test_tri_sum/2, deterministic)
        ],
        [test_tri_sum/2]
    ),
    compile_wam_predicate_to_go(test_tri_sum/2, "call test_tri_sum/2, 2",
        [foreign_lowering(ForeignSpec)], Code),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignNativeKind("test_tri_sum/2", "countdown_sum2")')),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignResultLayout("test_tri_sum/2", "tuple:1")')),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignResultMode("test_tri_sum/2", "deterministic")')),
    assertion(sub_string(Code, _, _, _, 'return vm.executeForeignPredicate("test_tri_sum", 2)')).

test(foreign_call_rewrite_generation) :-
    once((
        ForeignSpec = foreign_predicate(
            test_tri_sum/2,
            [ register_foreign_native_kind(test_tri_sum/2, countdown_sum2),
              register_foreign_result_layout(test_tri_sum/2, tuple(1)),
              register_foreign_result_mode(test_tri_sum/2, deterministic)
            ],
            [helper/2]
        ),
        wam_go_target:wam_line_to_go_literal(["call", "helper/2", "2"], test_tri_sum/2,
            [foreign_lowering(ForeignSpec)], Lit1),
        wam_go_target:wam_line_to_go_literal(["execute", "helper/2"], test_tri_sum/2,
            [foreign_lowering(ForeignSpec)], Lit2),
        assertion(Lit1 == '&CallForeign{Pred: "test_tri_sum", Arity: 2}'),
        assertion(Lit2 == '&CallForeign{Pred: "test_tri_sum", Arity: 2}')
    )).

test(foreign_auto_detect_generation) :-
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_reaches/2, "call test_reaches/2, 2",
        [foreign_lowering(true)], Code),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignNativeKind("test_reaches/2", "transitive_closure2")')),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignResultMode("test_reaches/2", "stream")')),
    assertion(sub_string(Code, _, _, _, 'vm.registerForeignStringConfig("test_reaches/2", "edge_pred", "edge/2")')),
    assertion(sub_string(Code, _, _, _, 'vm.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{{Left: "a", Right: "b"}, {Left: "b", Right: "c"}, {Left: "c", Right: "d"}})')),
    assertion(sub_string(Code, _, _, _, 'return vm.executeForeignPredicate("test_reaches", 2)')).

test(foreign_execution_deterministic_and_stream) :-
    once((
        get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_foreign_~w', [T]),
        TailSpec = foreign_predicate(
            test_tail_suffix/2,
            [ register_foreign_native_kind(test_tail_suffix/2, list_suffix2),
              register_foreign_result_layout(test_tail_suffix/2, tuple(1)),
              register_foreign_result_mode(test_tail_suffix/2, stream)
            ],
            [test_tail_suffix/2]
        ),
        write_wam_go_project(
            [plunit_wam_go_foreign_lowering:test_tail_suffix/2],
            [ module_name(go_foreign_test),
              foreign_lowering([TailSpec])
            ],
            TmpDir
        ),
        directory_file_path(TmpDir, 'foreign_test.go', TestPath),
        write_file(TestPath,
'package wam

import "testing"

func TestForeignCountdownSum(t *testing.T) {
    vm := NewWamState(nil, nil)
    vm.registerForeignNativeKind("test_tri_sum/2", "countdown_sum2")
    vm.registerForeignResultLayout("test_tri_sum/2", "tuple:1")
    vm.registerForeignResultMode("test_tri_sum/2", "deterministic")
    out := &Unbound{Name: "SUM"}
    vm.Regs["A1"] = &Integer{Val: 4}
    vm.Regs["A2"] = out
    if !vm.executeForeignPredicate("test_tri_sum", 2) {
        t.Fatalf("executeForeignPredicate failed")
    }
    got, ok := vm.deref(out).(*Integer)
    if !ok || got.Val != 10 {
        t.Fatalf("expected 10, got %#v", vm.deref(out))
    }
}

func TestForeignListSuffixStream(t *testing.T) {
    vm := NewWamState(nil, nil)
    vm.registerForeignNativeKind("test_tail_suffix/2", "list_suffix2")
    vm.registerForeignResultLayout("test_tail_suffix/2", "tuple:1")
    vm.registerForeignResultMode("test_tail_suffix/2", "stream")
    out := &Unbound{Name: "SUF"}
    vm.Regs["A1"] = &List{Elements: []Value{
        &Atom{Name: "a"},
        &Atom{Name: "b"},
    }}
    vm.Regs["A2"] = out
    if !vm.executeForeignPredicate("test_tail_suffix", 2) {
        t.Fatalf("executeForeignPredicate failed")
    }
    first, ok := vm.deref(out).(*List)
    if !ok || len(first.Elements) != 2 {
        t.Fatalf("expected first suffix [a,b], got %#v", vm.deref(out))
    }
    if !vm.backtrack() {
        t.Fatalf("expected stream backtrack result")
    }
    second, ok := vm.deref(out).(*List)
    if !ok || len(second.Elements) != 1 {
        t.Fatalf("expected second suffix [b], got %#v", vm.deref(out))
    }
    if !vm.backtrack() {
        t.Fatalf("expected final stream result")
    }
    third, ok := vm.deref(out).(*List)
    if !ok || len(third.Elements) != 0 {
        t.Fatalf("expected final suffix [], got %#v", vm.deref(out))
    }
}
'),
        (   catch(process_create(path(go), ['test', './...'], [cwd(TmpDir), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]), _, fail)
        ->  read_string(Out, _, _Stdout),
            read_string(Err, _, Stderr),
            process_wait(Pid, Exit),
            assertion(Exit == exit(0)),
            assertion(\+ sub_string(Stderr, _, _, _, 'FAIL'))
        ;   true
        ),
        delete_directory_and_contents(TmpDir)
    )).

:- end_tests(wam_go_foreign_lowering).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- begin_tests(wam_go_foreign_lowering).

:- dynamic test_tri_sum/2.
:- dynamic test_tail_suffix/2.
:- dynamic test_reaches/2.
:- dynamic test_reaches_dist/3.
:- dynamic test_parent_distance/4.
:- dynamic test_step_parent_distance/5.
:- dynamic test_weighted_path/3.
:- dynamic test_astar_weighted_path/4.
:- dynamic test_mixed_weighted_filtered/2.

edge(a, b).
edge(b, c).
edge(c, d).

weighted_edge(s, a, 1.0).
weighted_edge(s, b, 4.0).
weighted_edge(a, b, 2.0).
weighted_edge(a, c, 5.0).
weighted_edge(b, c, 1.0).
weighted_edge(c, d, 3.0).

user:direct_semantic_dist(s, a, 1.0).
user:direct_semantic_dist(s, b, 3.0).
user:direct_semantic_dist(s, c, 4.0).
user:direct_semantic_dist(s, d, 7.0).
user:direct_semantic_dist(a, b, 2.0).
user:direct_semantic_dist(a, c, 3.0).
user:direct_semantic_dist(a, d, 6.0).
user:direct_semantic_dist(b, c, 1.0).
user:direct_semantic_dist(b, d, 4.0).
user:direct_semantic_dist(c, d, 3.0).

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

test_reaches_dist(X, Y, 1) :- edge(X, Y).
test_reaches_dist(X, Y, D) :-
    edge(X, Z),
    test_reaches_dist(Z, Y, D1),
    D is D1 + 1.

test_parent_distance(X, Y, X, 1) :- edge(X, Y).
test_parent_distance(X, Y, Parent, D) :-
    edge(X, Z),
    test_parent_distance(Z, Y, Parent, D1),
    D is D1 + 1.

test_step_parent_distance(X, Y, Y, X, 1) :- edge(X, Y).
test_step_parent_distance(X, Y, Step, Parent, D) :-
    edge(X, Step),
    test_step_parent_distance(Step, Y, _Inner, Parent, D1),
    D is D1 + 1.

test_weighted_path(X, Y, W) :- weighted_edge(X, Y, W).
test_weighted_path(X, Y, Cost) :-
    weighted_edge(X, Z, W),
    test_weighted_path(Z, Y, RestCost),
    Cost is W + RestCost.

test_astar_weighted_path(X, Y, 5, W) :- weighted_edge(X, Y, W).
test_astar_weighted_path(X, Y, 5, Cost) :-
    weighted_edge(X, Z, W),
    test_astar_weighted_path(Z, Y, 5, RestCost),
    Cost is W + RestCost.

test_mixed_weighted_filtered(Target, Cost) :-
    test_weighted_path(s, Target, Cost),
    Cost < 5.0,
    atom(foo),
    \+ atom(5).

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

test(call_indexed_atom_fact2_literal) :-
    wam_go_target:wam_line_to_go_literal(["call_indexed_atom_fact2", "edge/2"], test_reaches/2,
        [], Lit),
    assertion(Lit == '&CallIndexedAtomFact2{Pred: "edge/2"}').

test(tsv_atom_fact2_setup_generation) :-
    wam_go_target:go_foreign_setup_line(register_tsv_atom_fact2(edge/2, '/tmp/edge.tsv'), Line),
    assertion(Line == '    if err := vm.registerTsvAtomFact2("edge/2", "/tmp/edge.tsv"); err != nil { panic(err) }').

test(lmdb_atom_fact2_setup_generation) :-
    wam_go_target:go_foreign_setup_line(register_lmdb_atom_fact2(edge/2, '/tmp/edge_artifact'), Line),
    assertion(Line == '    if err := vm.registerLmdbAtomFact2("edge/2", "/tmp/edge_artifact"); err != nil { panic(err) }').

test(atom_fact2_source_registry_runtime_shape) :-
    wam_go_target:compile_wam_runtime_to_go([], RuntimeCode),
    atom_string(RuntimeCode, Runtime),
    assertion(sub_string(Runtime, _, _, _, 'type staticAtomFact2Source struct')),
    assertion(sub_string(Runtime, _, _, _, 'func newStaticAtomFact2Source(pairs []AtomPair) *staticAtomFact2Source')),
    assertion(sub_string(Runtime, _, _, _, 'type lmdbAtomFact2Source struct')),
    assertion(sub_string(Runtime, _, _, _, 'func newLmdbAtomFact2Source(predKey string, artifactDir string) *lmdbAtomFact2Source')),
    assertion(sub_string(Runtime, _, _, _, 'func (vm *WamState) registerLmdbAtomFact2(predKey string, artifactDir string) error')),
    assertion(sub_string(Runtime, _, _, _, 'exec.Command(source.helperBin, args...).Output()')),
    assertion(sub_string(Runtime, _, _, _, 'func (vm *WamState) registerAtomFact2Source(predKey string, source AtomFact2Source)')),
    assertion(sub_string(Runtime, _, _, _, 'pairs = source.LookupArg1(key)')),
    read_file_to_string('templates/targets/go_wam/state.go.mustache', State, []),
    assertion(sub_string(State, _, _, _, 'type AtomFact2Source interface')),
    assertion(sub_string(State, _, _, _, 'AtomFact2Sources          map[string]AtomFact2Source')).

test(foreign_auto_detect_generation) :-
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_reaches/2, "call test_reaches/2, 2",
        [foreign_lowering(true)], ClosureCode),
    assertion(sub_string(ClosureCode, _, _, _, 'vm.registerForeignNativeKind("test_reaches/2", "transitive_closure2")')),
    assertion(sub_string(ClosureCode, _, _, _, 'vm.registerForeignStringConfig("test_reaches/2", "edge_pred", "edge/2")')),
    assertion(sub_string(ClosureCode, _, _, _, 'vm.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{{Left: "a", Right: "b"}, {Left: "b", Right: "c"}, {Left: "c", Right: "d"}})')),
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_reaches_dist/3, "call test_reaches_dist/3, 3",
        [foreign_lowering(true)], DistanceCode),
    assertion(sub_string(DistanceCode, _, _, _, 'vm.registerForeignNativeKind("test_reaches_dist/3", "transitive_distance3")')),
    assertion(sub_string(DistanceCode, _, _, _, 'vm.registerForeignResultLayout("test_reaches_dist/3", "tuple:2")')),
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_parent_distance/4, "call test_parent_distance/4, 4",
        [foreign_lowering(true)], ParentCode),
    assertion(sub_string(ParentCode, _, _, _, 'vm.registerForeignNativeKind("test_parent_distance/4", "transitive_parent_distance4")')),
    assertion(sub_string(ParentCode, _, _, _, 'vm.registerForeignResultLayout("test_parent_distance/4", "tuple:3")')),
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_step_parent_distance/5, "call test_step_parent_distance/5, 5",
        [foreign_lowering(true)], StepCode),
    assertion(sub_string(StepCode, _, _, _, 'vm.registerForeignNativeKind("test_step_parent_distance/5", "transitive_step_parent_distance5")')),
    assertion(sub_string(StepCode, _, _, _, 'vm.registerForeignResultLayout("test_step_parent_distance/5", "tuple:4")')),
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_weighted_path/3, "call test_weighted_path/3, 3",
        [foreign_lowering(true)], WeightedCode),
    assertion(sub_string(WeightedCode, _, _, _, 'vm.registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")')),
    assertion(sub_string(WeightedCode, _, _, _, 'vm.registerForeignStringConfig("test_weighted_path/3", "weight_pred", "weighted_edge/3")')),
    assertion(sub_string(WeightedCode, _, _, _, 'vm.registerIndexedWeightedEdgeTriples("weighted_edge/3", []WeightedEdgeTriple{')),
    compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_astar_weighted_path/4, "call test_astar_weighted_path/4, 4",
        [foreign_lowering(true)], AstarCode),
    assertion(sub_string(AstarCode, _, _, _, 'vm.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")')),
    assertion(sub_string(AstarCode, _, _, _, 'vm.registerForeignStringConfig("test_astar_weighted_path/4", "weight_pred", "weighted_edge/3")')),
    assertion(sub_string(AstarCode, _, _, _, 'vm.registerForeignUsizeConfig("test_astar_weighted_path/4", "dimensionality", 5)')),
    assertion(sub_string(AstarCode, _, _, _, 'return vm.executeForeignPredicate("test_astar_weighted_path", 4)')).

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
    out := &Unbound{Name: "SUM", Idx: 1}
    vm.Regs[0] = &Integer{Val: 4}
    vm.Regs[1] = out
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
    out := &Unbound{Name: "SUF", Idx: 1}
    vm.Regs[0] = &List{Elements: []Value{
        &Atom{Name: "a"},
        &Atom{Name: "b"},
    }}
    vm.Regs[1] = out
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

test(foreign_execution_graph_kernels) :-
    once((
        get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_graph_~w', [T]),
        write_wam_go_project(
            [plunit_wam_go_foreign_lowering:test_reaches/2],
            [module_name(go_graph_test)],
            TmpDir
        ),
        directory_file_path(TmpDir, 'foreign_graph_test.go', TestPath),
        write_file(TestPath,
'package wam

import (
    "os"
    "path/filepath"
    "testing"
)

func TestForeignGraphKernels(t *testing.T) {
    tsvPath := filepath.Join(t.TempDir(), "edge.tsv")
    if err := os.WriteFile(tsvPath, []byte("left\\tright\\nx\\ty\\nx\\tz\\n"), 0644); err != nil {
        t.Fatalf("write tsv: %v", err)
    }
    tsvVM := NewWamState([]Instruction{&CallIndexedAtomFact2{Pred: "tsv_edge/2"}, &Proceed{}}, map[string]int{})
    if err := tsvVM.registerTsvAtomFact2("tsv_edge/2", tsvPath); err != nil {
        t.Fatalf("register tsv facts: %v", err)
    }
    if _, ok := tsvVM.Ctx.AtomFact2Sources["tsv_edge/2"]; !ok {
        t.Fatalf("expected TSV source to register through AtomFact2Sources")
    }
    tsvTarget := &Unbound{Name: "TSV_TARGET", Idx: 1}
    tsvVM.Regs[0] = internAtom("x")
    tsvVM.Regs[1] = tsvTarget
    if !tsvVM.Run() {
        t.Fatalf("tsv-backed call_indexed_atom_fact2 failed")
    }
    if got, ok := tsvVM.deref(tsvTarget).(*Atom); !ok || got.Name != "y" {
        t.Fatalf("expected first tsv target y, got %#v", tsvVM.deref(tsvTarget))
    }
    if !tsvVM.backtrack() {
        t.Fatalf("expected second tsv fact result")
    }
    if got, ok := tsvVM.deref(tsvTarget).(*Atom); !ok || got.Name != "z" {
        t.Fatalf("expected second tsv target z, got %#v", tsvVM.deref(tsvTarget))
    }

    helperPath := filepath.Join(t.TempDir(), "lmdb_relation_artifact_mock")
    helperScript := "#!/bin/sh\\n" +
        "if [ \\"$1\\" = \\"get\\" ]; then\\n" +
        "  if [ \\"$4\\" = \\"x\\" ]; then printf \\"x\\\\ty\\\\nx\\\\tz\\\\n\\"; fi\\n" +
        "elif [ \\"$1\\" = \\"scan\\" ]; then\\n" +
        "  printf \\"x\\\\ty\\\\nx\\\\tz\\\\na\\\\tb\\\\n\\"\\n" +
        "fi\\n"
    if err := os.WriteFile(helperPath, []byte(helperScript), 0755); err != nil {
        t.Fatalf("write lmdb helper mock: %v", err)
    }
    t.Setenv("UW_LMDB_RELATION_ARTIFACT_BIN", helperPath)
    lmdbVM := NewWamState([]Instruction{&CallIndexedAtomFact2{Pred: "lmdb_edge/2"}, &Proceed{}}, map[string]int{})
    if err := lmdbVM.registerLmdbAtomFact2("lmdb_edge/2", filepath.Join(t.TempDir(), "edge_artifact")); err != nil {
        t.Fatalf("register lmdb facts: %v", err)
    }
    if _, ok := lmdbVM.Ctx.AtomFact2Sources["lmdb_edge/2"]; !ok {
        t.Fatalf("expected LMDB source to register through AtomFact2Sources")
    }
    lmdbTarget := &Unbound{Name: "LMDB_TARGET", Idx: 1}
    lmdbVM.Regs[0] = internAtom("x")
    lmdbVM.Regs[1] = lmdbTarget
    if !lmdbVM.Run() {
        t.Fatalf("lmdb-backed call_indexed_atom_fact2 failed")
    }
    if got, ok := lmdbVM.deref(lmdbTarget).(*Atom); !ok || got.Name != "y" {
        t.Fatalf("expected first lmdb target y, got %#v", lmdbVM.deref(lmdbTarget))
    }
    if !lmdbVM.backtrack() {
        t.Fatalf("expected second lmdb fact result")
    }
    if got, ok := lmdbVM.deref(lmdbTarget).(*Atom); !ok || got.Name != "z" {
        t.Fatalf("expected second lmdb target z, got %#v", lmdbVM.deref(lmdbTarget))
    }

    factVM := NewWamState([]Instruction{&CallIndexedAtomFact2{Pred: "edge/2"}, &Proceed{}}, map[string]int{})
    factVM.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{
        {Left: "a", Right: "b"},
        {Left: "a", Right: "c"},
        {Left: "b", Right: "d"},
    })
    if _, ok := factVM.Ctx.AtomFact2Sources["edge/2"]; !ok {
        t.Fatalf("expected indexed facts to register through AtomFact2Sources")
    }
    factTarget := &Unbound{Name: "FACT_TARGET", Idx: 1}
    factVM.Regs[0] = internAtom("a")
    factVM.Regs[1] = factTarget
    if !factVM.Run() {
        t.Fatalf("call_indexed_atom_fact2 failed")
    }
    if got, ok := factVM.deref(factTarget).(*Atom); !ok || got.Name != "b" {
        t.Fatalf("expected first indexed fact target b, got %#v", factVM.deref(factTarget))
    }
    if !factVM.backtrack() {
        t.Fatalf("expected second indexed fact result")
    }
    if got, ok := factVM.deref(factTarget).(*Atom); !ok || got.Name != "c" {
        t.Fatalf("expected second indexed fact target c, got %#v", factVM.deref(factTarget))
    }
    if factVM.backtrack() {
        t.Fatalf("expected indexed fact stream to be exhausted")
    }

    factExact := NewWamState([]Instruction{&CallIndexedAtomFact2{Pred: "edge/2"}, &Proceed{}}, map[string]int{})
    factExact.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{
        {Left: "a", Right: "b"},
        {Left: "a", Right: "c"},
    })
    factExact.Regs[0] = internAtom("a")
    factExact.Regs[1] = internAtom("c")
    if !factExact.Run() {
        t.Fatalf("expected exact indexed fact match a-c")
    }

    factFail := NewWamState([]Instruction{&CallIndexedAtomFact2{Pred: "edge/2"}, &Proceed{}}, map[string]int{})
    factFail.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{{Left: "a", Right: "b"}})
    factFail.Regs[0] = internAtom("missing")
    factFail.Regs[1] = &Unbound{Name: "NO_FACT", Idx: 1}
    if factFail.Run() {
        t.Fatalf("expected indexed fact lookup for missing key to fail")
    }

    vm := NewWamState(nil, nil)
    vm.registerForeignNativeKind("test_reaches_dist/3", "transitive_distance3")
    vm.registerForeignResultLayout("test_reaches_dist/3", "tuple:2")
    vm.registerForeignResultMode("test_reaches_dist/3", "stream")
    vm.registerForeignStringConfig("test_reaches_dist/3", "edge_pred", "edge/2")
    vm.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{
        {Left: "a", Right: "b"},
        {Left: "b", Right: "c"},
        {Left: "c", Right: "d"},
    })
    target := &Unbound{Name: "TARGET", Idx: 1}
    dist := &Unbound{Name: "DIST", Idx: 2}
    vm.Regs[0] = &Atom{Name: "a"}
    vm.Regs[1] = target
    vm.Regs[2] = dist
    if !vm.executeForeignPredicate("test_reaches_dist", 3) {
        t.Fatalf("transitive_distance3 failed")
    }
    if got, ok := vm.deref(target).(*Atom); !ok || got.Name != "b" {
        t.Fatalf("expected first target b, got %#v", vm.deref(target))
    }
    if got, ok := vm.deref(dist).(*Integer); !ok || got.Val != 1 {
        t.Fatalf("expected first distance 1, got %#v", vm.deref(dist))
    }
    if !vm.backtrack() {
        t.Fatalf("expected second transitive_distance3 result")
    }
    if got, ok := vm.deref(target).(*Atom); !ok || got.Name != "c" {
        t.Fatalf("expected second target c, got %#v", vm.deref(target))
    }
    if got, ok := vm.deref(dist).(*Integer); !ok || got.Val != 2 {
        t.Fatalf("expected second distance 2, got %#v", vm.deref(dist))
    }

    vm2 := NewWamState(nil, nil)
    vm2.registerForeignNativeKind("test_parent_distance/4", "transitive_parent_distance4")
    vm2.registerForeignResultLayout("test_parent_distance/4", "tuple:3")
    vm2.registerForeignResultMode("test_parent_distance/4", "stream")
    vm2.registerForeignStringConfig("test_parent_distance/4", "edge_pred", "edge/2")
    vm2.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{
        {Left: "a", Right: "b"},
        {Left: "b", Right: "c"},
        {Left: "c", Right: "d"},
    })
    pTarget := &Unbound{Name: "PTARGET", Idx: 1}
    pParent := &Unbound{Name: "PPARENT", Idx: 2}
    pDist := &Unbound{Name: "PDIST", Idx: 3}
    vm2.Regs[0] = &Atom{Name: "a"}
    vm2.Regs[1] = pTarget
    vm2.Regs[2] = pParent
    vm2.Regs[3] = pDist
    if !vm2.executeForeignPredicate("test_parent_distance", 4) {
        t.Fatalf("transitive_parent_distance4 failed")
    }
    if got, ok := vm2.deref(pTarget).(*Atom); !ok || got.Name != "b" {
        t.Fatalf("expected parent-distance target b, got %#v", vm2.deref(pTarget))
    }
    if got, ok := vm2.deref(pParent).(*Atom); !ok || got.Name != "a" {
        t.Fatalf("expected parent-distance parent a, got %#v", vm2.deref(pParent))
    }

    vm3 := NewWamState(nil, nil)
    vm3.registerForeignNativeKind("test_step_parent_distance/5", "transitive_step_parent_distance5")
    vm3.registerForeignResultLayout("test_step_parent_distance/5", "tuple:4")
    vm3.registerForeignResultMode("test_step_parent_distance/5", "stream")
    vm3.registerForeignStringConfig("test_step_parent_distance/5", "edge_pred", "edge/2")
    vm3.registerIndexedAtomFact2Pairs("edge/2", []AtomPair{
        {Left: "a", Right: "b"},
        {Left: "b", Right: "c"},
        {Left: "c", Right: "d"},
    })
    sTarget := &Unbound{Name: "STARGET", Idx: 1}
    sStep := &Unbound{Name: "SSTEP", Idx: 2}
    sParent := &Unbound{Name: "SPARENT", Idx: 3}
    sDist := &Unbound{Name: "SDIST", Idx: 4}
    vm3.Regs[0] = &Atom{Name: "a"}
    vm3.Regs[1] = sTarget
    vm3.Regs[2] = sStep
    vm3.Regs[3] = sParent
    vm3.Regs[4] = sDist
    if !vm3.executeForeignPredicate("test_step_parent_distance", 5) {
        t.Fatalf("transitive_step_parent_distance5 failed")
    }
    if got, ok := vm3.deref(sStep).(*Atom); !ok || got.Name != "b" {
        t.Fatalf("expected step b, got %#v", vm3.deref(sStep))
    }

    weighted := []WeightedEdgeTriple{
        {Left: "s", Right: "a", Weight: 1.0},
        {Left: "s", Right: "b", Weight: 4.0},
        {Left: "a", Right: "b", Weight: 2.0},
        {Left: "a", Right: "c", Weight: 5.0},
        {Left: "b", Right: "c", Weight: 1.0},
        {Left: "c", Right: "d", Weight: 3.0},
    }
    vm4 := NewWamState(nil, nil)
    vm4.registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")
    vm4.registerForeignResultLayout("test_weighted_path/3", "tuple:2")
    vm4.registerForeignResultMode("test_weighted_path/3", "stream")
    vm4.registerForeignStringConfig("test_weighted_path/3", "weight_pred", "weighted_edge/3")
    vm4.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    wTarget := &Unbound{Name: "WTARGET", Idx: 1}
    wCost := &Unbound{Name: "WCOST", Idx: 2}
    vm4.Regs[0] = &Atom{Name: "s"}
    vm4.Regs[1] = wTarget
    vm4.Regs[2] = wCost
    if !vm4.executeForeignPredicate("test_weighted_path", 3) {
        t.Fatalf("weighted_shortest_path3 failed")
    }
    weightedResults := make([]string, 0)
    if gotT, okT := vm4.deref(wTarget).(*Atom); okT {
        if gotC, okC := vm4.deref(wCost).(*Float); okC {
            weightedResults = append(weightedResults, gotT.Name+":"+gotC.String())
        } else {
            t.Fatalf("expected first weighted cost float, got %#v", vm4.deref(wCost))
        }
    } else {
        t.Fatalf("expected first weighted target atom, got %#v", vm4.deref(wTarget))
    }
    for vm4.backtrack() {
        gotT, okT := vm4.deref(wTarget).(*Atom)
        gotC, okC := vm4.deref(wCost).(*Float)
        if !okT || !okC {
            t.Fatalf("expected weighted backtrack tuple, got %#v %#v", vm4.deref(wTarget), vm4.deref(wCost))
        }
        weightedResults = append(weightedResults, gotT.Name+":"+gotC.String())
    }
    expectedWeighted := []string{"a:1", "b:3", "c:4", "d:7"}
    if len(weightedResults) != len(expectedWeighted) {
        t.Fatalf("expected %d weighted results, got %#v", len(expectedWeighted), weightedResults)
    }
    for i := range expectedWeighted {
        if weightedResults[i] != expectedWeighted[i] {
            t.Fatalf("expected weighted results %v, got %v", expectedWeighted, weightedResults)
        }
    }

    vm4Exact := NewWamState(nil, nil)
    vm4Exact.registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")
    vm4Exact.registerForeignResultLayout("test_weighted_path/3", "tuple:2")
    vm4Exact.registerForeignResultMode("test_weighted_path/3", "stream")
    vm4Exact.registerForeignStringConfig("test_weighted_path/3", "weight_pred", "weighted_edge/3")
    vm4Exact.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    vm4Exact.Regs[0] = &Atom{Name: "s"}
    vm4Exact.Regs[1] = &Atom{Name: "d"}
    vm4Exact.Regs[2] = &Float{Val: 7.0}
    if !vm4Exact.executeForeignPredicate("test_weighted_path", 3) {
        t.Fatalf("expected weighted exact match for s->d cost 7.0")
    }

    vm4Fail := NewWamState(nil, nil)
    vm4Fail.registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")
    vm4Fail.registerForeignResultLayout("test_weighted_path/3", "tuple:2")
    vm4Fail.registerForeignResultMode("test_weighted_path/3", "stream")
    vm4Fail.registerForeignStringConfig("test_weighted_path/3", "weight_pred", "weighted_edge/3")
    vm4Fail.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    vm4Fail.Regs[0] = &Atom{Name: "d"}
    vm4Fail.Regs[1] = &Unbound{Name: "WTFAIL", Idx: 1}
    vm4Fail.Regs[2] = &Unbound{Name: "WCFAIL", Idx: 2}
    if vm4Fail.executeForeignPredicate("test_weighted_path", 3) {
        t.Fatalf("expected weighted path from d to fail")
    }

    heuristic := []WeightedEdgeTriple{
        {Left: "s", Right: "a", Weight: 1.0},
        {Left: "s", Right: "b", Weight: 3.0},
        {Left: "s", Right: "c", Weight: 4.0},
        {Left: "s", Right: "d", Weight: 7.0},
        {Left: "a", Right: "b", Weight: 2.0},
        {Left: "a", Right: "c", Weight: 3.0},
        {Left: "a", Right: "d", Weight: 6.0},
        {Left: "b", Right: "c", Weight: 1.0},
        {Left: "b", Right: "d", Weight: 4.0},
        {Left: "c", Right: "d", Weight: 3.0},
    }
    vm5 := NewWamState(nil, nil)
    vm5.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")
    vm5.registerForeignResultLayout("test_astar_weighted_path/4", "tuple:1")
    vm5.registerForeignResultMode("test_astar_weighted_path/4", "stream")
    vm5.registerForeignStringConfig("test_astar_weighted_path/4", "weight_pred", "weighted_edge/3")
    vm5.registerForeignStringConfig("test_astar_weighted_path/4", "direct_dist_pred", "direct_semantic_dist/3")
    vm5.registerForeignUsizeConfig("test_astar_weighted_path/4", "dimensionality", 5)
    vm5.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    vm5.registerIndexedWeightedEdgeTriples("direct_semantic_dist/3", heuristic)
    aCost := &Unbound{Name: "ACOST", Idx: 3}
    vm5.Regs[0] = &Atom{Name: "s"}
    vm5.Regs[1] = &Atom{Name: "d"}
    vm5.Regs[2] = &Integer{Val: 5}
    vm5.Regs[3] = aCost
    if !vm5.executeForeignPredicate("test_astar_weighted_path", 4) {
        t.Fatalf("astar_shortest_path4 failed")
    }
    if got, ok := vm5.deref(aCost).(*Float); !ok || got.Val != 7.0 {
        t.Fatalf("expected astar cost 7.0, got %#v", vm5.deref(aCost))
    }
    if vm5.backtrack() {
        t.Fatalf("expected astar_shortest_path4 single result")
    }

    vm5Fallback := NewWamState(nil, nil)
    vm5Fallback.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")
    vm5Fallback.registerForeignResultLayout("test_astar_weighted_path/4", "tuple:1")
    vm5Fallback.registerForeignResultMode("test_astar_weighted_path/4", "stream")
    vm5Fallback.registerForeignStringConfig("test_astar_weighted_path/4", "weight_pred", "weighted_edge/3")
    vm5Fallback.registerForeignUsizeConfig("test_astar_weighted_path/4", "dimensionality", 5)
    vm5Fallback.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    fallbackCost := &Unbound{Name: "FALLBACK_COST", Idx: 3}
    vm5Fallback.Regs[0] = &Atom{Name: "s"}
    vm5Fallback.Regs[1] = &Atom{Name: "d"}
    vm5Fallback.Regs[2] = &Integer{Val: 5}
    vm5Fallback.Regs[3] = fallbackCost
    if !vm5Fallback.executeForeignPredicate("test_astar_weighted_path", 4) {
        t.Fatalf("expected astar fallback without heuristic to succeed")
    }
    if got, ok := vm5Fallback.deref(fallbackCost).(*Float); !ok || got.Val != 7.0 {
        t.Fatalf("expected astar fallback cost 7.0, got %#v", vm5Fallback.deref(fallbackCost))
    }

    vm5Exact := NewWamState(nil, nil)
    vm5Exact.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")
    vm5Exact.registerForeignResultLayout("test_astar_weighted_path/4", "tuple:1")
    vm5Exact.registerForeignResultMode("test_astar_weighted_path/4", "stream")
    vm5Exact.registerForeignStringConfig("test_astar_weighted_path/4", "weight_pred", "weighted_edge/3")
    vm5Exact.registerForeignStringConfig("test_astar_weighted_path/4", "direct_dist_pred", "direct_semantic_dist/3")
    vm5Exact.registerForeignUsizeConfig("test_astar_weighted_path/4", "dimensionality", 5)
    vm5Exact.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    vm5Exact.registerIndexedWeightedEdgeTriples("direct_semantic_dist/3", heuristic)
    vm5Exact.Regs[0] = &Atom{Name: "s"}
    vm5Exact.Regs[1] = &Atom{Name: "d"}
    vm5Exact.Regs[2] = &Integer{Val: 5}
    vm5Exact.Regs[3] = &Float{Val: 7.0}
    if !vm5Exact.executeForeignPredicate("test_astar_weighted_path", 4) {
        t.Fatalf("expected exact astar match for s->d cost 7.0")
    }

    vm5Fail := NewWamState(nil, nil)
    vm5Fail.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")
    vm5Fail.registerForeignResultLayout("test_astar_weighted_path/4", "tuple:1")
    vm5Fail.registerForeignResultMode("test_astar_weighted_path/4", "stream")
    vm5Fail.registerForeignStringConfig("test_astar_weighted_path/4", "weight_pred", "weighted_edge/3")
    vm5Fail.registerForeignUsizeConfig("test_astar_weighted_path/4", "dimensionality", 5)
    vm5Fail.registerIndexedWeightedEdgeTriples("weighted_edge/3", weighted)
    vm5Fail.Regs[0] = &Atom{Name: "d"}
    vm5Fail.Regs[1] = &Atom{Name: "s"}
    vm5Fail.Regs[2] = &Integer{Val: 5}
    vm5Fail.Regs[3] = &Unbound{Name: "NO_ASTAR", Idx: 3}
    if vm5Fail.executeForeignPredicate("test_astar_weighted_path", 4) {
        t.Fatalf("expected astar path from d to s to fail")
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

test(foreign_project_auto_detect_weighted_and_astar) :-
    once((
        get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_project_foreign_~w', [T]),
        write_wam_go_project(
            [plunit_wam_go_foreign_lowering:test_weighted_path/3,
             plunit_wam_go_foreign_lowering:test_astar_weighted_path/4],
            [module_name(go_project_foreign_test),
             foreign_lowering(true)],
            TmpDir
        ),
        directory_file_path(TmpDir, 'lib.go', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        assertion(sub_string(LibCode, _, _, _, 'func Test_weighted_path(a1 Value, a2 Value, a3 Value) bool {')),
        assertion(sub_string(LibCode, _, _, _, 'func Test_astar_weighted_path(a1 Value, a2 Value, a3 Value, a4 Value) bool {')),
        assertion(sub_string(LibCode, _, _, _, 'vm.registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")')),
        assertion(sub_string(LibCode, _, _, _, 'vm.registerForeignNativeKind("test_astar_weighted_path/4", "astar_shortest_path4")')),
        directory_file_path(TmpDir, 'foreign_project_test.go', TestPath),
        write_file(TestPath,
'package wam

import "testing"

func TestForeignProjectAutoDetectWeightedAndAstar(t *testing.T) {
    if !Test_weighted_path(&Atom{Name: "s"}, &Atom{Name: "d"}, &Float{Val: 7.0}) {
        t.Fatalf("expected auto-detected weighted exact match to succeed")
    }
    if Test_weighted_path(&Atom{Name: "d"}, &Atom{Name: "s"}, &Unbound{Name: "WEIGHTED_FAIL", Idx: 2}) {
        t.Fatalf("expected weighted project reverse query to fail")
    }

    if !Test_astar_weighted_path(&Atom{Name: "s"}, &Atom{Name: "d"}, &Integer{Val: 5}, &Float{Val: 7.0}) {
        t.Fatalf("expected auto-detected astar exact match to succeed")
    }
    if Test_astar_weighted_path(&Atom{Name: "d"}, &Atom{Name: "s"}, &Integer{Val: 5}, &Unbound{Name: "ASTAR_FAIL", Idx: 3}) {
        t.Fatalf("expected astar project reverse query to fail")
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

test(foreign_mixed_boundary_weighted_backtracking) :-
    once((
        get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_mixed_foreign_~w', [T]),
        write_wam_go_project(
            [plunit_wam_go_foreign_lowering:test_mixed_weighted_filtered/2,
             plunit_wam_go_foreign_lowering:test_weighted_path/3],
            [module_name(go_mixed_foreign_test),
             foreign_lowering(true)],
            TmpDir
        ),
        directory_file_path(TmpDir, 'lib.go', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        assertion(sub_string(LibCode, _, _, _, '&CallForeign{Pred: "test_weighted_path", Arity: 3}')),
        directory_file_path(TmpDir, 'mixed_foreign_test.go', TestPath),
        write_file(TestPath,
'package wam

import "testing"

func TestMixedBoundaryWeightedBacktracking(t *testing.T) {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = Test_mixed_weighted_filteredStartPC
    target := &Unbound{Name: "TARGET", Idx: 0}
    cost := &Unbound{Name: "COST", Idx: 1}
    vm.Regs[0] = target
    vm.Regs[1] = cost
    if !vm.Run() {
        t.Fatalf("expected mixed weighted caller to succeed")
    }
    gotTarget, okTarget := vm.deref(target).(*Atom)
    gotCost, okCost := vm.deref(cost).(*Float)
    if !okTarget || !okCost || gotTarget.Name != "a" || gotCost.Val != 1.0 {
        t.Fatalf("expected first mixed result a/1.0, got %#v %#v", vm.deref(target), vm.deref(cost))
    }
    if !vm.backtrack() {
        t.Fatalf("expected second mixed result")
    }
    if !vm.Run() {
        t.Fatalf("expected mixed caller to resume after second backtrack")
    }
    gotTarget, okTarget = vm.deref(target).(*Atom)
    gotCost, okCost = vm.deref(cost).(*Float)
    if !okTarget || !okCost || gotTarget.Name != "b" || gotCost.Val != 3.0 {
        t.Fatalf("expected second mixed result b/3.0, got %#v %#v", vm.deref(target), vm.deref(cost))
    }
    if !vm.backtrack() {
        t.Fatalf("expected third mixed result")
    }
    if !vm.Run() {
        t.Fatalf("expected mixed caller to resume after third backtrack")
    }
    gotTarget, okTarget = vm.deref(target).(*Atom)
    gotCost, okCost = vm.deref(cost).(*Float)
    if !okTarget || !okCost || gotTarget.Name != "c" || gotCost.Val != 4.0 {
        t.Fatalf("expected third mixed result c/4.0, got %#v %#v", vm.deref(target), vm.deref(cost))
    }
    if !vm.backtrack() {
        t.Fatalf("expected final foreign backtrack to reach filtered-out result")
    }
    if vm.Run() {
        t.Fatalf("expected mixed weighted caller to stop after filtered results")
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

test(foreign_no_kernels_suppresses_go_auto_detection) :-
    once((
        compile_wam_predicate_to_go(plunit_wam_go_foreign_lowering:test_weighted_path/3,
            "call test_weighted_path/3, 3",
            [foreign_lowering(true), no_kernels(true)],
            Code),
        assertion(\+ sub_string(Code, _, _, _, 'registerForeignNativeKind("test_weighted_path/3", "weighted_shortest_path3")')),
        assertion(\+ sub_string(Code, _, _, _, 'executeForeignPredicate("test_weighted_path", 3)')),
        assertion(sub_string(Code, _, _, _, '&Call{Pred: "test_weighted_path/3", Arity: 3}'))
    )).

:- end_tests(wam_go_foreign_lowering).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

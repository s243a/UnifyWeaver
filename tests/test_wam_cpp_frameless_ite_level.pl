:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_cpp_frameless_ite_level.pl
%
% Probe for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write* form
% on wam_cpp. Ported from tests/test_wam_python_frameless_ite_level.pl (ledger
% row D52) and tests/test_wam_go_frameless_ite_level.pl (row D53), which port
% the wam_rust finding of row D50.
%
% The shape
% ---------
% `compile_if_then_else/7` in the shared emitter (`wam_target.pl`) reserves a
% permanent Y register for the if-then-else barrier AFTER it has decided
% whether the clause needs an environment. So a clause that needs no
% environment still gets `get_level Yn` ... `cut Yn` -- with NO `allocate`.
% `sat/2` clause 2 below is exactly that clause (`\+ G` inlines to
% `(G -> fail ; true)` under `ite_use_y_level(true)`, which EVERY wam_cpp
% compile enables -- wam_cpp_target.pl:439,1425,2641), and `pick_a/4` /
% `pick_b/4` are callers that DO hold an environment with live Y registers
% across the call.
%
% wam_cpp routes every Y-register access to `env_stack.back().y_regs`
% (WamState::get_cell / set_cell / put_reg). For a frameless callee that
% back-frame is the CALLER's, so the old `get_level Y1` handler --
% `bind_cell(get_cell("Y1"), Integer(choice_points.size()))` -- wrote a
% choice-point depth straight over the caller's permanent variable Y1.
% On the pristine tree the interpreter lane gave:
%
%   pick_a(3, gte(1), tagX, Out)  ->  FAILED       (SWI: Out = tagX)
%   pick_b(3, gte(1), tagX, Out)  ->  Out = 0      (a CP depth; SWI: Out = tagX)
%   wpick(4, tagY, Out)           ->  FAILED       (SWI: Out = tagY)
%
% i.e. a silent wrong answer, not a crash.
%
% The fix keeps ITE barrier levels on the if-then-else's own choice point
% (`ChoicePoint::levels` + `record_ite_level`/`lookup_ite_level` in the C++
% runtime, with a pending-level park so a `get_level` emitted immediately
% before `try_me_else` lands on the guard CP the try_me_else creates), so the
% level never touches a register and is per-activation for free -- the
% wam_rust `ChoicePoint::levels` / wam_python / wam_go model.
%
% Two entry lanes
% ---------------
%   * INTERPRETER lane -- the predicate reached at its WAM label via
%     WamState::query / run(). Broken on pristine, fixed here.
%   * LOWERED lane -- a `lowered_pick_a_4(WamState*)` function (emit_mode
%     functions). Its `allocate` gives pick_a a real env frame and it holds
%     Y1/Y2, then dispatches `call sat/2` through the interpreter
%     (`vm->run()` at the label). So the callee's `get_level` runs the same
%     runtime handler; the CP-levels fix covers this lane too. (The pure
%     lowered-structural ITE emits `get_level`/`cut` as no-ops -- the commit
%     is a C++ if/else -- so it never wrote a Y barrier in the first place.)
%
% This suite pins BOTH halves:
%   * emission -- `sat/2` really does carry `get_level` in an `allocate`-less
%     clause (if the shared emitter ever stops doing that the probe would go
%     vacuously green, so we assert the shape), and the lowered functions
%     really do exist (so the exposed lowered lane cannot silently vanish);
%   * behaviour -- the generated C++ project agrees with SWI as oracle on
%     BOTH lanes, including a recursive caller that keeps several activations
%     of the same if-then-else live.
%
% Skipped automatically when no C++ compiler is available.
%
%   swipl -q -g run_tests -t halt tests/test_wam_cpp_frameless_ite_level.pl

:- module(test_wam_cpp_frameless_ite_level,
          [test_wam_cpp_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target',
              [write_wam_cpp_project/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4, user:wpick/3.

% --- the probe program (also the SWI oracle) -------------------------------

user:lt(A, B) :- A < B.

% Multi-clause callee. Clause 2 needs NO environment: its only permanent
% would be the if-then-else barrier the emitter reserves for the inlined
% negation -- so it emits `get_level Y1` with no `allocate`.
user:sat(_V, any).
user:sat(V, gte(G)) :- \+ user:lt(V, G).

% Callers that DO hold an environment across the call. The Y numbering
% follows first use in the body, so pick_a parks the (unbound) output in Y1
% and pick_b parks the (bound) input tag in Y1 -- the first shape used to
% FAIL, the second used to return the clobbering choice-point index.
user:pick_a(Ver, C, Tag, Out) :- user:sat(Ver, C), Out = Tag.
user:pick_b(Ver, C, Tag, Out) :- user:sat(Ver, C), Tag = Out.

% Recursive caller: several activations of sat/2's if-then-else are live at
% once, each with its own barrier. A register-held level is not
% per-activation; a level on the guard choice point is.
user:wpick(0, Tag, Tag).
user:wpick(N, Tag, Out) :-
    N > 0,
    user:sat(N, gte(1)),
    M is N - 1,
    user:wpick(M, Tag, Out).

probe_preds([user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4,
             user:wpick/3]).

% Lowered C++ functions that must exist for the lowered lane to be exercised.
% If lowering ever stops covering these the execution arm would still pass
% while testing nothing, so their absence is a loud failure.
lowered_function('lowered_pick_a_4').
lowered_function('lowered_pick_b_4').

% --- oracle cases ----------------------------------------------------------
%
% probe_case(Id, Label, LoweredFn, Args, OutReg, Goal, OutVar)
%   LoweredFn : the lowered C++ function, or `none` to run only the
%               interpreter lane.
%   Args      : the full argument list (an `out` placeholder marks the
%               fresh output variable's position).
%   OutReg    : the A-register the answer lands in (e.g. 'A4'), or `none`
%               for a success/failure-only case.
%   OutVar    : the SWI variable holding the answer, or 0 for success/fail.

probe_case(sat_any,     'sat/2',    none,
           [int(3), atom(any)], none,
           user:sat(3, any), 0).
probe_case(sat_gte_t,   'sat/2',    none,
           [int(3), gte(1)], none,
           user:sat(3, gte(1)), 0).
probe_case(sat_gte_f,   'sat/2',    none,
           [int(0), gte(1)], none,
           user:sat(0, gte(1)), 0).
probe_case(pick_a_true, 'pick_a/4', 'lowered_pick_a_4',
           [int(3), gte(1), atom(tagX), out], 'A4',
           user:pick_a(3, gte(1), tagX, O), O).
probe_case(pick_b_true, 'pick_b/4', 'lowered_pick_b_4',
           [int(3), gte(1), atom(tagX), out], 'A4',
           user:pick_b(3, gte(1), tagX, O), O).
probe_case(pick_a_fail, 'pick_a/4', 'lowered_pick_a_4',
           [int(0), gte(1), atom(tagX), out], 'A4',
           user:pick_a(0, gte(1), tagX, _), 0).
probe_case(pick_b_fail, 'pick_b/4', 'lowered_pick_b_4',
           [int(0), gte(1), atom(tagX), out], 'A4',
           user:pick_b(0, gte(1), tagX, _), 0).
probe_case(pick_a_any,  'pick_a/4', 'lowered_pick_a_4',
           [int(3), atom(any), atom(tagX), out], 'A4',
           user:pick_a(3, any, tagX, O), O).
probe_case(wpick_deep,  'wpick/3',  none,
           [int(4), atom(tagY), out], 'A3',
           user:wpick(4, tagY, O), O).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ).
cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

test_wam_cpp_frameless_ite_level :-
    run_tests(wam_cpp_frameless_ite_level).

:- begin_tests(wam_cpp_frameless_ite_level).

% The probe is only meaningful while the shared emitter still produces the
% hazard shape. Assert it directly on the emitted text.
test(sat_clause_has_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% ... and while the callers really do park a permanent across the call.
test(caller_allocates_and_holds_y1) :-
    compile_predicate_to_wam_text(user:pick_b/4, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "allocate")),
    assertion(sub_string(S, _, _, _, "Y1")),
    assertion(sub_string(S, _, _, _, "call sat/2")).

test(matches_swi_on_both_lanes, [condition(cpp_compiler(_))]) :-
    once(run_frameless_probe).

:- end_tests(wam_cpp_frameless_ite_level).

% --- harness ---------------------------------------------------------------

run_frameless_probe :-
    cpp_compiler(CC),
    probe_preds(Preds),
    Dir = 'output/test_wam_cpp_frameless_ite_level',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % emit_mode(functions) gives BOTH the interpreter labels and the lowered
    % functions in one build, so a single harness drives both lanes.
    write_wam_cpp_project(Preds,
                          [module_name(cfprobe), emit_mode(functions),
                           wam_fallback(true)], Dir),
    atomic_list_concat([Dir, '/cpp'], CppDir),
    % Confirm the lowered lane is really covered before trusting it.
    atomic_list_concat([CppDir, '/generated_program.cpp'], GenPath),
    read_file_to_string(GenPath, GenSrc, []),
    forall(lowered_function(F),
           (   sub_string(GenSrc, _, _, _, F)
           ->  true
           ;   format(user_error,
                      'frameless-ITE probe: generated_program.cpp has no ~w \c
                       -- the lowered entry lane is no longer covered~n', [F]),
               fail
           )),
    probe_cpp_source(Src),
    atomic_list_concat([CppDir, '/frameless_probe.cpp'], TestPath),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    format(atom(Cmd),
        '~w -std=c++17 -O0 ~w/frameless_probe.cpp ~w/generated_program.cpp \c
         ~w/wam_runtime.cpp -o ~w/frameless_probe 2>&1 && ~w/frameless_probe',
        [CC, CppDir, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0), sub_string(OutStr, _, _, _, "ALL PASS")
    ->  true
    ;   format(user_error,
               "~n[frameless-ITE-level harness output]~n~w~n", [OutStr]),
        throw(wam_cpp_frameless_ite_level_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

probe_cpp_source(Src) :-
    findall(Proto, lowered_prototype(Proto), Protos),
    atomic_list_concat(Protos, '\n', ProtoBlock),
    findall(Line, case_check_line(Line), Lines),
    atomic_list_concat(Lines, '\n', ChecksBlock),
    cpp_prelude(Prelude),
    cpp_main_open(MainOpen),
    cpp_epilogue(Epilogue),
    atomic_list_concat([Prelude, ProtoBlock, '\n', MainOpen,
                        ChecksBlock, Epilogue], '\n', Src).

lowered_prototype(Proto) :-
    lowered_function(F),
    format(atom(Proto), 'bool ~w(WamState*);', [F]).

% One chk(...) per lane per case, with SWI supplying the expectation.
case_check_line(Line) :-
    probe_case(Id, Label, Fn, Args, OutReg, Goal, OutVar),
    maplist(cpp_arg, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgList),
    (   catch(Goal, _, fail)
    ->  (   OutVar == 0
        ->  Expect = 'true'
        ;   cpp_answer_literal(OutVar, Expect)
        )
    ;   Expect = 'FAIL'
    ),
    ( OutReg == none -> OutRegS = '' ; OutRegS = OutReg ),
    (   Fn == none
    ->  format(atom(Line),
               "    chk(\"~w/interp\", runInterp(\"~w\", {~w}, \"~w\"), \"~w\");",
               [Id, Label, ArgList, OutRegS, Expect])
    ;   format(atom(Line),
               "    chk(\"~w/interp\", runInterp(\"~w\", {~w}, \"~w\"), \"~w\");\n\c
                    chk(\"~w/lowered\", runLowered(&~w, {~w}, \"~w\"), \"~w\");",
               [Id, Label, ArgList, OutRegS, Expect,
                Id, Fn, ArgList, OutRegS, Expect])
    ).

% The harness reduces an answer to a bare atom name or decimal integer, so
% the expectation SWI computes is spelled the same way.
cpp_answer_literal(V, S) :- number(V), !, format(atom(S), '~w', [V]).
cpp_answer_literal(V, S) :- format(atom(S), '~w', [V]).

cpp_arg(int(N), S)  :- format(atom(S), 'I(~w)', [N]).
cpp_arg(atom(A), S) :- format(atom(S), 'A("~w")', [A]).
cpp_arg(gte(N), S)  :- format(atom(S), 'gte(~w)', [N]).
cpp_arg(out, 'U()').

cpp_prelude(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
// Auto-generated frameless-ITE-level harness (do not edit).
#include \"wam_runtime.h\"
#include <iostream>
#include <string>
#include <vector>

static Value I(long long n) { return Value::Integer(n); }
static Value A(const char* s) { return Value::Atom(s); }
static Value U() { return Value::Unbound(\"_Out\"); }
static Value gte(long long n) {
    std::vector<wam_cpp::CellPtr> a;
    a.push_back(std::make_shared<Value>(I(n)));
    return Value::Compound(\"gte/1\", std::move(a));
}

// Reduce a solution to a comparable string: the atom name, the decimal
// integer, \"UNBOUND\" for a success that left the output unbound, or \"FAIL\".
static std::string answer(bool ok, WamState& vm, const std::string& reg) {
    if (!ok) return \"FAIL\";
    if (reg.empty()) return \"true\";
    Value v = vm.deref(vm.get_reg(reg));
    switch (v.tag) {
        case Value::Tag::Atom:    return v.s;
        case Value::Tag::Integer: return std::to_string(v.i);
        case Value::Tag::Unbound:
        case Value::Tag::Uninit:  return \"UNBOUND\";
        default:                  return \"OTHER\";
    }
}

// Lane 1: the interpreter, entered at the predicate label via query().
static std::string runInterp(const char* label, std::vector<Value> args,
                             const std::string& reg) {
    WamState vm; Program::apply_setup(vm);
    bool ok = vm.query(label, args);
    return answer(ok, vm, reg);
}

// Lane 2: the lowered C++ function. Its `allocate` gives the caller a real
// env frame holding the Y registers the callee's frameless `get_level` used
// to scribble over; `call sat/2` dispatches through the interpreter, so the
// CP-levels fix covers this lane too.
static std::string runLowered(bool (*fn)(WamState*), std::vector<Value> args,
                              const std::string& reg) {
    WamState vm; Program::apply_setup(vm);
    for (auto& c : vm.regs) c.reset();
    for (std::size_t k = 0; k < args.size(); ++k)
        vm.set_cell(\"A\" + std::to_string(k + 1),
                    std::make_shared<Value>(args[k]));
    bool ok = fn(&vm);
    return answer(ok, vm, reg);
}

static int fails = 0;
static void chk(const char* name, const std::string& got,
                const std::string& want) {
    if (got != want) {
        fails++;
        std::cout << \"FAIL \" << name << \" got \" << got
                  << \" want \" << want << \"\\n\";
    }
}
").

cpp_main_open("int main() {").

cpp_epilogue(
"    if (fails == 0) { std::cout << \"ALL PASS\\n\"; return 0; }
    std::cout << \"FAILURES: \" << fails << \"\\n\"; return 1;
}").

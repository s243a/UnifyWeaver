% test_wam_lua_lowered_t4.pl
%
% End-to-end execution test for the Lua T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go/C++/Haskell/F#/Clojure.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers to ALL clauses inline: each clause is an
% immediately-invoked closure, tried in order with a trail/register restore
% (state.regs copy + trail unwind) between attempts. The first clause that
% succeeds wins (first-solution / deterministic-prefix); the predicate never
% falls back to the bytecode interpreter, unlike multi_clause_1.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade_r/2 — RULE chain with a REPEATED first-arg constant (alice in
%                 clauses 1 & 3), so it is not a distinct-first-arg (T5) chain;
%   * rel/2     — RULE chain with a VARIABLE first arg (=/2 body; needs the
%                 switch_on_constant_a2 prefix skipped to reach try_me_else).
% (Fact-only predicates are routed to the Lua target's inline fact-table
% optimisation, not the lowered function, so the T4 path is exercised with
% rule clauses.)
%
% Skipped automatically when no Lua interpreter is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_lua_target').
:- use_module('../src/unifyweaver/targets/wam_lua_lowered_emitter').

:- dynamic user:grade_r/2.
:- dynamic user:rel/2.

% A RULE chain (not facts): fact-only predicates are routed to the Lua
% target's inline fact-table optimisation instead of the lowered function,
% so the T4 path is exercised with rule clauses. grade_r mirrors grade but
% binds the second argument in the body, with a REPEATED first-arg constant
% (alice in clauses 1 & 3) so it is not a distinct-first-arg (T5) chain.
user:grade_r(alice, G) :- G = a.
user:grade_r(bob,   G) :- G = b.
user:grade_r(alice, G) :- G = c.

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

lua_interpreter(Exe) :-
    member(Exe, ['lua5.4', 'lua5.3', 'lua', 'luajit']),
    catch(( process_create(path(Exe), ['-v'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, _) ), _, fail), !.

:- begin_tests(wam_lua_lowered_t4, [condition(lua_interpreter(_))]).

test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade_r/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_lua_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    lua_interpreter(Lua),
    Dir = 'output/test_wam_lua_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_lua_project(
        [user:grade_r/2, user:rel/2],
        [module_name('t4proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/lua/generated_program.lua'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_grade_r_2", "lowered_rel_2", "T4 all-clauses inline",
                      "return lowered_grade_r_2", "return lowered_rel_2"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    atomic_list_concat([Dir, '/lua/harness.lua'], HPath),
    harness_source_t4(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    atomic_list_concat([Dir, '/lua'], LuaDir),
    format(atom(Cmd), 'cd ~w && ~w harness.lua 2>&1', [LuaDir, Lua]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[lua t4 test output]~n~w~n", [OutStr]),
        throw(lua_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_lua_lowered_t4).

% Calls each per-predicate module entry with a BOUND first argument and checks
% the boolean. Exercises the non-first clauses (grade clauses 2 & 3, rel clause
% 2) — the T4 payoff — plus no-match cases. Atoms are built through the shared
% intern table so the ids match the lowered functions.
harness_source_t4(
"package.path = \"./?.lua;\" .. package.path
local M = require(\"generated_program\")
local R = M.Runtime
local function A(s) return M.V.Atom(R.intern(M.program.intern_table, s)) end
local fails, total = 0, 0
local function chk(name, got, want)
  total = total + 1
  if got ~= want then
    fails = fails + 1
    print(\"FAIL\", name, \"got\", tostring(got), \"want\", tostring(want))
  end
end
chk(\"grade_r(alice,a)\", M.grade_r(A(\"alice\"), A(\"a\")), true)
chk(\"grade_r(bob,b)\",   M.grade_r(A(\"bob\"),   A(\"b\")), true)
chk(\"grade_r(alice,c)\", M.grade_r(A(\"alice\"), A(\"c\")), true)
chk(\"grade_r(alice,b)\", M.grade_r(A(\"alice\"), A(\"b\")), false)
chk(\"grade_r(carol,a)\", M.grade_r(A(\"carol\"), A(\"a\")), false)
chk(\"grade_r(bob,c)\",   M.grade_r(A(\"bob\"),   A(\"c\")), false)
chk(\"rel(p,one)\", M.rel(A(\"p\"), A(\"one\")), true)
chk(\"rel(q,two)\", M.rel(A(\"q\"), A(\"two\")), true)
chk(\"rel(p,two)\", M.rel(A(\"p\"), A(\"two\")), false)
chk(\"rel(q,one)\", M.rel(A(\"q\"), A(\"one\")), false)
if fails == 0 then print(\"ALL \" .. total .. \" PASS\") else print(fails .. \" FAILURES\") end
os.exit(fails == 0 and 0 or 1)
").

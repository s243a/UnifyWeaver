% test_wam_lua_lowered_t5.pl
%
% End-to-end execution test for the Lua T5 lowering — "multi-clause as a
% first-argument dispatch" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md). Lua is line-based, so
% the shared wam_clause_chain front-end is fed a minimal term view of the
% tokenized lines (separators + head get_constant + opaque line(Parts) leaves);
% emission renders each clause's remainder back through the line emitter.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument constant
% now lowers to a bound-checked first-arg dispatch over ALL clauses (non-first
% clauses become fast-path too when A1 is bound), instead of multi_clause_1
% (clause 1 inline, clauses 2+ via the array fallback). An unbound first
% argument falls back to that same multi_clause_1 path, which enumerates.
%
% Pins (the harness preloads a BOUND first arg, exercising every clause incl.
% the non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * pick/2  — RULE chain; each remainder binds the second argument via =/2,
%               exercising a non-trivial multi-instruction clause body.
%
% Two predicate shapes are deliberately NOT used here, for Lua-specific
% reasons unrelated to T5:
%   - arity-2 fact chains (e.g. sz(small,1)) are compiled to an indexed FACT
%     STREAM by wam_lua_target and never reach the lowered function;
%   - is/2 over a built compound (e.g. R is 1+1) is unsupported by the Lua
%     runtime's number_value (it evaluates only int/float, not a +/2 struct),
%     so such a clause fails in the interpreter too.
% A rule chain with =/2 remainders (pick/2) covers the multi-instruction
% clause body without hitting either.
%
% Skipped automatically when no Lua interpreter is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_lua_target').
:- use_module('../src/unifyweaver/targets/wam_lua_lowered_emitter').

:- dynamic user:color/1.
:- dynamic user:pick/2.

user:color(red).
user:color(green).
user:color(blue).

user:pick(a, X) :- X = apple.
user:pick(b, X) :- X = banana.
user:pick(c, X) :- X = cherry.

lua_interpreter(Exe) :-
    member(Exe, ['lua5.4', 'lua5.3', 'lua', 'luajit']),
    catch(( process_create(path(Exe), ['-e', 'os.exit(0)'],
                           [stdin(null), stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail), !.

:- begin_tests(wam_lua_lowered_t5, [condition(lua_interpreter(_))]).

% Both predicates must lower as T5 (clause_chain), not multi_clause_1.
test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, pick/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_lua_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    lua_interpreter(Lua),
    Dir = 'output/test_wam_lua_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the lowered Lua project.
    write_wam_lua_project(
        [user:color/1, user:pick/2],
        [module_name('t5proj'), emit_mode(functions)], Dir),
    % Sanity: the predicates must be lowered as the T5 dispatch and routed
    % through the lowered function (not the array/fact-stream fallback).
    atomic_list_concat([Dir, '/lua/generated_program.lua'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_color_1", "lowered_pick_2",
                      "T5 first-argument dispatch",
                      "return lowered_color_1", "return lowered_pick_2"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    % 2. Write the harness next to the generated module.
    atomic_list_concat([Dir, '/lua/harness.lua'], HPath),
    harness_source(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    % 3. Run it.
    atomic_list_concat([Dir, '/lua'], LuaDir),
    format(atom(Cmd), 'cd ~w && ~w harness.lua 2>&1', [LuaDir, Lua]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 9 PASS")
    ->  true
    ;   format(user_error, "~n[lua t5 test output]~n~w~n", [OutStr]),
        throw(lua_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_lua_lowered_t5).

% Calls each per-predicate module entry with a BOUND first argument and checks
% the boolean. color exercises atom discriminators incl. the non-first clauses
% (green/blue) and a no-match (yellow). pick exercises a multi-instruction
% remainder: the matching clause binds/checks the second argument via =/2, so
% pick(a,banana) is false (clause a yields apple) while pick(b,banana) is true
% (the non-first clause runs natively — the T5 payoff). Atoms are built through
% the shared intern table so the ids match the lowered functions.
harness_source(
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
chk(\"color(red)\",    M.color(A(\"red\")),    true)
chk(\"color(green)\",  M.color(A(\"green\")),  true)
chk(\"color(blue)\",   M.color(A(\"blue\")),   true)
chk(\"color(yellow)\", M.color(A(\"yellow\")), false)
chk(\"pick(a,apple)\",  M.pick(A(\"a\"), A(\"apple\")),  true)
chk(\"pick(a,banana)\", M.pick(A(\"a\"), A(\"banana\")), false)
chk(\"pick(b,banana)\", M.pick(A(\"b\"), A(\"banana\")), true)
chk(\"pick(c,cherry)\", M.pick(A(\"c\"), A(\"cherry\")), true)
chk(\"pick(d,x)\",      M.pick(A(\"d\"), A(\"x\")),      false)
if fails == 0 then print(\"ALL \" .. total .. \" PASS\") else print(fails .. \" FAILURES\") end
os.exit(fails == 0 and 0 or 1)
").

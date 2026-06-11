% test_wam_lua_lowered_t6.pl
%
% End-to-end test for the Lua T6 lowering — first-argument indexing via a hash
% table keyed by the interned atom id, lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% Lua is int-interned: an atom is {tag="atom", id=<int>}, so the T5 cascade is
% `if t5a1.tag=="atom" and t5a1.id==n then …`. T6 replaces the linear cascade
% with a dispatch table `_t6[t5a1.id]` of per-clause closures, built ONCE at
% module load inside a `do` block (no per-call allocation). Benchmarked 1.7x at
% 8, 8.2x at 64, 29.7x at 256 (interpreted Lua: the table lookup is O(1), the
% if-cascade O(n)).
%
% Gated like Rust/C++/F#/Go: T6 fires only when every clause discriminates on a
% distinct ATOM and there are >= t6_min_clauses of them (default 8). Below the
% threshold the few-clause predicate stays the T5 cascade.
%
% Skipped automatically when no lua interpreter is on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_lua_target').
:- use_module('../src/unifyweaver/targets/wam_lua_lowered_emitter').

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

lua_interpreter(Exe) :-
    member(Exe, ['lua5.4', 'lua5.3', 'lua', 'luajit']),
    catch(( process_create(path(Exe), ['-v'], [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, _) ), _, fail), !.

:- begin_tests(wam_lua_lowered_t6, [condition(lua_interpreter(_))]).

% Codegen gate: shade/grade (>=8 atoms) emit the table dispatch (T6); few stays
% the cascade (T5). Threshold configurable.
test(gate_picks_t6_for_many_t5_for_few) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    lower_predicate_to_lua(shade/1, Ws, [], lowered(_, _, ShadeCode)),
    assertion(sub_string(ShadeCode, _, _, _, "T6 first-argument indexing")),
    assertion(sub_string(ShadeCode, _, _, _, "_t6[t5a1.id]")),
    wam_target:compile_predicate_to_wam(grade/2, [], Wg),
    lower_predicate_to_lua(grade/2, Wg, [], lowered(_, _, GradeCode)),
    assertion(sub_string(GradeCode, _, _, _, "T6 first-argument indexing")),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_lua(few/1, Wf, [], lowered(_, _, FewCode)),
    assertion(\+ sub_string(FewCode, _, _, _, "T6 first-argument indexing")),
    lower_predicate_to_lua(few/1, Wf, [t6_min_clauses(3)], lowered(_, _, FewT6)),
    assertion(sub_string(FewT6, _, _, _, "T6 first-argument indexing")).

% Build + run: the T6 table dispatch returns the Prolog-correct result for
% clause hits (incl. non-first clauses) and a no-match.
test(t6_exec) :-
    lua_interpreter(Lua),
    Dir = 'output/test_wam_lua_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_lua_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/lua/generated_program.lua'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    assertion(sub_string(ProgSrc, _, _, _, "T6 first-argument indexing")),
    atomic_list_concat([Dir, '/lua/harness.lua'], HPath),
    lua_t6_harness(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/lua'], LuaDir),
    format(atom(Cmd), 'cd ~w && ~w harness.lua 2>&1', [LuaDir, Lua]),
    process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 8 PASS")
    ->  true
    ;   format(user_error, "~n[lua t6 test output]~n~w~n", [OutStr]),
        throw(lua_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_lua_lowered_t6).

lua_t6_harness(
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
chk(\"shade(s01)\", M.shade(A(\"s01\")), true)
chk(\"shade(s05)\", M.shade(A(\"s05\")), true)
chk(\"shade(s10)\", M.shade(A(\"s10\")), true)
chk(\"shade(zz)\",  M.shade(A(\"zz\")),  false)
chk(\"few(a)\",     M.few(A(\"a\")),     true)
chk(\"few(b)\",     M.few(A(\"b\")),     true)
chk(\"few(c)\",     M.few(A(\"c\")),     true)
chk(\"few(z)\",     M.few(A(\"z\")),     false)
if fails == 0 then print(\"ALL \" .. total .. \" PASS\") else print(fails .. \" FAILURES\") end
os.exit(fails == 0 and 0 or 1)
").

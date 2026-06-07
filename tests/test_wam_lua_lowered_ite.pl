% test_wam_lua_lowered_ite.pl
%
% End-to-end execution test for WAM-Lua if-then-else / negation / once
% lowering (emit_mode(functions)). Counterpart to the Go / Rust / C++ /
% Haskell / F# / Clojure / LLVM / Elixir / Python lowered-ITE exec tests.
% Pins:
%
%   * simple ITE         — lite/2;
%   * negation (\+)       — lneg/1 (commit is the !/0 builtin: then = fail/0,
%                          else = true/0, run after a trail rollback);
%   * sequential ITEs    — lseqite/3 (two sibling blocks);
%   * nested ITEs         — lnestite/2 (inner block in the then-arm).
%
% Lua previously declined any predicate whose lowered body contained the
% soft-cut block's try_me_else / cut_ite / jump / trust_me (they were not in
% parts_supported/1), so such predicates fell back to the WAM interpreter —
% sound but not lowered. Folding clause 1 through wam_ite_structurer emits
% native Lua if/else. lua's bind_var always trails, so undoing the trail to
% the pre-condition mark before the else restores the condition's bindings.
%
% Generates a lowered Lua project and calls the per-predicate module entry
% points (M.lite, M.lneg, ...) asserting the boolean outcome. Skipped unless
% a Lua interpreter (lua5.4 / lua5.3 / lua / luajit) is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_lua_target').

:- dynamic user:lite/2.
:- dynamic user:lneg/1.
:- dynamic user:lseqite/3.
:- dynamic user:lnestite/2.

user:lite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:lneg(X)          :- \+ X > 0.
user:lseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:lnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

lua_interpreter(Exe) :-
    member(Exe, ['lua5.4', 'lua5.3', 'lua', 'luajit']),
    % Use -e 'os.exit(0)' (not -v): a bare `lua -v` with no script enters
    % the interactive REPL and blocks reading stdin. Redirect stdin to null
    % as belt-and-braces.
    catch(( process_create(path(Exe), ['-e', 'os.exit(0)'],
                           [stdin(null), stdout(null), stderr(null),
                            process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail), !.

:- begin_tests(wam_lua_lowered_ite, [condition(lua_interpreter(_))]).

test(ite_exec_parity) :-
    lua_interpreter(Lua),
    Dir = 'output/test_wam_lua_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the lowered Lua project.
    write_wam_lua_project(
        [user:lite/2, user:lneg/1, user:lseqite/3, user:lnestite/2],
        [module_name('iteproj'), emit_mode(functions)], Dir),
    % Sanity: the predicates must actually be lowered as native if/else
    % (not the interpreter fallback), else the test would pass vacuously.
    atomic_list_concat([Dir, '/lua/generated_program.lua'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_lite_2", "lowered_lneg_1",
                      "lowered_lseqite_3", "lowered_lnestite_2"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    assertion(sub_string(ProgSrc, _, _, _, "if-then-else / negation / once")),
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
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[lua ite test output]~n~w~n", [OutStr]),
        throw(lua_ite_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_lua_lowered_ite).

% Calls each per-predicate module entry. Atoms are built through the shared
% intern table (Runtime.intern), so the ids match the constants the lowered
% functions compare against.
harness_source(
"package.path = \"./?.lua;\" .. package.path
local M = require(\"generated_program\")
local R = M.Runtime
local function A(s) return M.V.Atom(R.intern(M.program.intern_table, s)) end
local function I(n) return M.V.Int(n) end
local fails, total = 0, 0
local function chk(name, got, want)
  total = total + 1
  if got ~= want then
    fails = fails + 1
    print(\"FAIL\", name, \"got\", tostring(got), \"want\", tostring(want))
  end
end
chk(\"lite(5,pos)\",     M.lite(I(5),  A(\"pos\")),    true)
chk(\"lite(5,nonpos)\",  M.lite(I(5),  A(\"nonpos\")), false)
chk(\"lite(-1,nonpos)\", M.lite(I(-1), A(\"nonpos\")), true)
chk(\"lite(-1,pos)\",    M.lite(I(-1), A(\"pos\")),    false)
chk(\"lneg(5)\",  M.lneg(I(5)),  false)
chk(\"lneg(-1)\", M.lneg(I(-1)), true)
chk(\"lneg(0)\",  M.lneg(I(0)),  true)
chk(\"lseqite(10,pos,big)\",      M.lseqite(I(10), A(\"pos\"),    A(\"big\")),   true)
chk(\"lseqite(10,pos,small)\",    M.lseqite(I(10), A(\"pos\"),    A(\"small\")), false)
chk(\"lseqite(3,pos,small)\",     M.lseqite(I(3),  A(\"pos\"),    A(\"small\")), true)
chk(\"lseqite(-1,nonpos,small)\", M.lseqite(I(-1), A(\"nonpos\"), A(\"small\")), true)
chk(\"lnestite(20,big)\",   M.lnestite(I(20), A(\"big\")),   true)
chk(\"lnestite(5,small)\",  M.lnestite(I(5),  A(\"small\")), true)
chk(\"lnestite(-1,neg)\",   M.lnestite(I(-1), A(\"neg\")),   true)
chk(\"lnestite(20,small)\", M.lnestite(I(20), A(\"small\")), false)
if fails == 0 then print(\"ALL \" .. total .. \" PASS\") else print(fails .. \" FAILURES\") end
").

% test_wam_cpp_lowered_ite_exec.pl
%
% End-to-end execution test for C++ if-then-else / negation / once lowering
% (emit_mode(functions)).
%
% Generates a WAM C++ project with the lowered emitter enabled, compiles it
% with the host C++17 compiler, and runs a harness that calls each lowered
% function and asserts the boolean result. Counterpart to the Go and Rust
% tests. Pins:
%
%   * sequential ITEs   — cseqite(10,pos,small) must be false;
%   * nested ITEs       — cnestite/2;
%   * negation (\+)     — cneg/1 (commits with !/0);
%   * binding condition — cundoite/2, whose condition binds a fresh var then
%                         fails; needs the trail unwind AND shared variable
%                         identity (put_variable aliasing one cell).
%
% Before this fix the lowered emitter no-op'd try_me_else/cut_ite/jump/
% trust_me, dropping the structure entirely. C++ also compiles ITE with
% ite_use_y_level(true), so the commit is `cut Yn` (not cut_ite) — handled
% by the shared structurer's is_commit/1. Skipped when no C++ compiler.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target').

:- dynamic user:cite/2.
:- dynamic user:cneg/1.
:- dynamic user:cseqite/3.
:- dynamic user:cnestite/2.
:- dynamic user:cundoite/2.

user:cite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:cneg(X)          :- \+ X > 0.
user:cseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:cnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).
user:cundoite(X, R)   :- ( (Y = a, Y = b) -> R = then ; R = els ), X = Y.

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ).
cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_cpp_lowered_ite_exec, [condition(cpp_compiler(_))]).

test(ite_exec_parity) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM C++ project with the lowered emitter enabled.
    write_wam_cpp_project(
        [user:cite/2, user:cneg/1, user:cseqite/3, user:cnestite/2, user:cundoite/2],
        [module_name('itecpp'), wam_fallback(true), emit_mode(functions)], Dir),
    % 2. Test harness alongside the generated sources.
    atomic_list_concat([Dir, '/cpp/test_ite.cpp'], TestPath),
    cpp_test_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    % 3. Compile + run.
    atomic_list_concat([Dir, '/cpp'], CppDir),
    format(atom(Cmd),
        '~w -std=c++17 -O0 ~w/test_ite.cpp ~w/generated_program.cpp ~w/wam_runtime.cpp -o ~w/ite_test 2>&1 && ~w/ite_test',
        [CC, CppDir, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 17 PASS")
    ->  true
    ;   format(user_error, "~n[cpp ite test output]~n~w~n", [OutStr]),
        throw(cpp_test_failed(Status))
    ).

:- end_tests(wam_cpp_lowered_ite_exec).

% Calls each lowered function with the query args preloaded into the
% A-registers and asserts the boolean outcome. cseqite(10,pos,small)=false
% and cundoite(c,els)=true are the discriminating cases.
cpp_test_source(
"#include \"wam_runtime.h\"
#include <iostream>
bool lowered_cite_2(WamState*); bool lowered_cneg_1(WamState*); bool lowered_cseqite_3(WamState*);
bool lowered_cnestite_2(WamState*); bool lowered_cundoite_2(WamState*);
static int failures = 0;
static void chk(const char* n, bool g, bool w) {
    if (g != w) { std::cerr << \"FAIL \" << n << \": got \" << g << \" want \" << w << \"\\n\"; failures++; }
}
static Value I(long long n) { return Value::Integer(n); }
static Value A(const char* s) { return Value::Atom(s); }
int main() {
    { WamState v; v.put_reg(\"A1\", I(5));  v.put_reg(\"A2\", A(\"pos\"));    chk(\"cite(5,pos)\",    lowered_cite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(5));  v.put_reg(\"A2\", A(\"nonpos\")); chk(\"cite(5,nonpos)\", lowered_cite_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", I(-1)); v.put_reg(\"A2\", A(\"nonpos\")); chk(\"cite(-1,nonpos)\",lowered_cite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(-1)); v.put_reg(\"A2\", A(\"pos\"));    chk(\"cite(-1,pos)\",   lowered_cite_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", I(5));  chk(\"cneg(5)\",  lowered_cneg_1(&v), false); }
    { WamState v; v.put_reg(\"A1\", I(-1)); chk(\"cneg(-1)\", lowered_cneg_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(0));  chk(\"cneg(0)\",  lowered_cneg_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(10)); v.put_reg(\"A2\", A(\"pos\")); v.put_reg(\"A3\", A(\"big\"));   chk(\"cseqite(10,pos,big)\",   lowered_cseqite_3(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(10)); v.put_reg(\"A2\", A(\"pos\")); v.put_reg(\"A3\", A(\"small\")); chk(\"cseqite(10,pos,small)\", lowered_cseqite_3(&v), false); }
    { WamState v; v.put_reg(\"A1\", I(3));  v.put_reg(\"A2\", A(\"pos\")); v.put_reg(\"A3\", A(\"small\")); chk(\"cseqite(3,pos,small)\",  lowered_cseqite_3(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(-1)); v.put_reg(\"A2\", A(\"nonpos\")); v.put_reg(\"A3\", A(\"small\")); chk(\"cseqite(-1,nonpos,small)\", lowered_cseqite_3(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(20)); v.put_reg(\"A2\", A(\"big\"));   chk(\"cnestite(20,big)\",   lowered_cnestite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(5));  v.put_reg(\"A2\", A(\"small\")); chk(\"cnestite(5,small)\",  lowered_cnestite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(-1)); v.put_reg(\"A2\", A(\"neg\"));   chk(\"cnestite(-1,neg)\",   lowered_cnestite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", I(20)); v.put_reg(\"A2\", A(\"small\")); chk(\"cnestite(20,small)\", lowered_cnestite_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"c\")); v.put_reg(\"A2\", A(\"els\"));  chk(\"cundoite(c,els)\",  lowered_cundoite_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"c\")); v.put_reg(\"A2\", A(\"then\")); chk(\"cundoite(c,then)\", lowered_cundoite_2(&v), false); }
    if (failures == 0) { std::cout << \"ALL 17 PASS\\n\"; return 0; }
    std::cerr << failures << \" FAILURES\\n\"; return 1;
}
").

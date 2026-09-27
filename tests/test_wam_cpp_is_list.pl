:- encoding(utf8).
% test_wam_cpp_is_list.pl — Focused test suite for C++ WAM is_list/1 builtin.
%
% Validates that is_list/1:
% - Dereferences its argument
% - Succeeds for finite proper lists: [] and normal [|]/2 cons lists
% - Rejects improper lists and improper chains
% - Rejects open lists (unbound tail) and open chains
% - Rejects cyclic lists (self-loops) and cyclic chains
% - Rejects non-list atoms, integers, compounds, and unbound variables
% - Preserves choicepoint, trail, and caller-continuation behavior

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target').

:- dynamic user:cpp_list_nil/1.
:- dynamic user:cpp_list_normal/1.
:- dynamic user:cpp_list_cons/1.
:- dynamic user:cpp_list_improper/1.
:- dynamic user:cpp_list_improper_chain/1.
:- dynamic user:cpp_list_open/1.
:- dynamic user:cpp_list_open_chain/1.
:- dynamic user:cpp_list_cyclic/1.
:- dynamic user:cpp_list_cyclic_chain/1.
:- dynamic user:cpp_list_atom/1.
:- dynamic user:cpp_list_var/1.
:- dynamic user:cpp_list_int/1.
:- dynamic user:cpp_list_compound/1.
:- dynamic user:cpp_list_continuation_success/1.
:- dynamic user:cpp_list_continuation_fail/1.
:- dynamic user:cpp_list_continuation_cyclic/1.
:- dynamic user:cpp_list_arg/1.

% 1. Empty list []
user:cpp_list_nil(R) :-
    ( is_list([]) -> R is 1 ; R is 0 ).

% 2. Normal list
user:cpp_list_normal(R) :-
    ( is_list([1, 2, 3]) -> R is 1 ; R is 0 ).

% 3. Compiled [|]/2 list
user:cpp_list_cons(R) :-
    L = '[|]'(1, '[|]'(2, [])),
    ( is_list(L) -> R is 1 ; R is 0 ).

% 4. Improper list (short: [1|2])
user:cpp_list_improper(R) :-
    ( is_list('[|]'(1, 2)) -> R is 1 ; R is 0 ).

% 4b. Improper list chain ([1, 2, 3 | 4])
user:cpp_list_improper_chain(R) :-
    L = '[|]'(1, '[|]'(2, '[|]'(3, 4))),
    ( is_list(L) -> R is 1 ; R is 0 ).

% 5. Open list (short: [1|Tail])
user:cpp_list_open(R) :-
    L = '[|]'(1, _Tail),
    ( is_list(L) -> R is 1 ; R is 0 ).

% 5b. Open list chain ([1, 2 | Tail])
user:cpp_list_open_chain(R) :-
    L = '[|]'(1, '[|]'(2, _Tail)),
    ( is_list(L) -> R is 1 ; R is 0 ).

% 6. Cyclic list (self-loop: X = [1|X])
user:cpp_list_cyclic(R) :-
    X = '[|]'(1, X),
    ( is_list(X) -> R is 1 ; R is 0 ).

% 6b. Cyclic list chain (3-element cycle: X = [1, 2, 3 | X])
user:cpp_list_cyclic_chain(R) :-
    X = '[|]'(1, '[|]'(2, '[|]'(3, X))),
    ( is_list(X) -> R is 1 ; R is 0 ).

% 7. Non-list atom
user:cpp_list_atom(R) :-
    ( is_list(foo) -> R is 1 ; R is 0 ).

% 8. Unbound argument
user:cpp_list_var(R) :-
    ( is_list(_X) -> R is 1 ; R is 0 ).

% 8b. Integer and compound
user:cpp_list_int(R) :-
    ( is_list(42) -> R is 1 ; R is 0 ).
user:cpp_list_compound(R) :-
    ( is_list(foo(1, 2)) -> R is 1 ; R is 0 ).

% 9. Continuation check: on success, execution continues to subsequent goals
user:cpp_list_continuation_success(R) :-
    is_list([1, 2, 3]),
    R is 1.

% 10. Continuation check: on failure, subsequent goals in clause do not run,
% and backtracking cleanly proceeds to the next clause
user:cpp_list_continuation_fail(R) :-
    is_list('[|]'(1, 2)),
    R is 0.
user:cpp_list_continuation_fail(1).

% 11. Continuation check with cyclic failure: fails cleanly, does not hang,
% and backtracks to next clause
user:cpp_list_continuation_cyclic(R) :-
    X = '[|]'(1, X),
    is_list(X),
    R is 0.
user:cpp_list_continuation_cyclic(1).

% 12. Direct CLI argument passing
user:cpp_list_arg(X) :- is_list(X).

gpp_available :-
    catch(( process_create(path('g++'), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_cpp_is_list, [condition(gpp_available)]).

test(is_list_exec_coverage) :-
    Dir = 'output/test_wam_cpp_is_list',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_cpp_project(
        [user:cpp_list_nil/1,
         user:cpp_list_normal/1,
         user:cpp_list_cons/1,
         user:cpp_list_improper/1,
         user:cpp_list_improper_chain/1,
         user:cpp_list_open/1,
         user:cpp_list_open_chain/1,
         user:cpp_list_cyclic/1,
         user:cpp_list_cyclic_chain/1,
         user:cpp_list_atom/1,
         user:cpp_list_var/1,
         user:cpp_list_int/1,
         user:cpp_list_compound/1,
         user:cpp_list_continuation_success/1,
         user:cpp_list_continuation_fail/1,
         user:cpp_list_continuation_cyclic/1,
         user:cpp_list_arg/1],
        [module_name(cpp_is_list), emit_main(true)], Dir),
    atomic_list_concat([Dir, '/cpp'], CppDir),
    atomic_list_concat([CppDir, '/is_list_bin'], ExePath),
    format(atom(BuildCmd),
        'g++ -std=c++17 -O0 -o ~w ~w/*.cpp 2>&1', [ExePath, CppDir]),
    shell_ok(BuildCmd),
    Cases = [
        % [] succeeds
        'cpp_list_nil/1'-1-true,
        'cpp_list_nil/1'-0-false,
        % normal list succeeds
        'cpp_list_normal/1'-1-true,
        'cpp_list_normal/1'-0-false,
        % compiled [|]/2 list succeeds
        'cpp_list_cons/1'-1-true,
        'cpp_list_cons/1'-0-false,
        % improper list fails
        'cpp_list_improper/1'-0-true,
        'cpp_list_improper/1'-1-false,
        % improper chain fails
        'cpp_list_improper_chain/1'-0-true,
        'cpp_list_improper_chain/1'-1-false,
        % open list fails
        'cpp_list_open/1'-0-true,
        'cpp_list_open/1'-1-false,
        % open chain fails
        'cpp_list_open_chain/1'-0-true,
        'cpp_list_open_chain/1'-1-false,
        % cyclic list fails
        'cpp_list_cyclic/1'-0-true,
        'cpp_list_cyclic/1'-1-false,
        % cyclic chain fails
        'cpp_list_cyclic_chain/1'-0-true,
        'cpp_list_cyclic_chain/1'-1-false,
        % atom fails
        'cpp_list_atom/1'-0-true,
        'cpp_list_atom/1'-1-false,
        % unbound var fails
        'cpp_list_var/1'-0-true,
        'cpp_list_var/1'-1-false,
        % int & compound fail
        'cpp_list_int/1'-0-true,
        'cpp_list_int/1'-1-false,
        'cpp_list_compound/1'-0-true,
        'cpp_list_compound/1'-1-false,
        % continuation checks
        'cpp_list_continuation_success/1'-1-true,
        'cpp_list_continuation_success/1'-0-false,
        'cpp_list_continuation_fail/1'-1-true,
        'cpp_list_continuation_fail/1'-0-false,
        'cpp_list_continuation_cyclic/1'-1-true,
        'cpp_list_continuation_cyclic/1'-0-false,
        % CLI arg queries
        'cpp_list_arg/1'-"[]"-true,
        'cpp_list_arg/1'-"[1,2,3]"-true,
        'cpp_list_arg/1'-foo-false,
        'cpp_list_arg/1'-42-false,
        'cpp_list_arg/1'-"box(1,2)"-false
    ],
    forall(member(Pred-Arg-Expect, Cases),
           run_case(ExePath, Pred, Arg, Expect)),
    delete_directory_and_contents(Dir).

:- end_tests(wam_cpp_is_list).

shell_ok(Cmd) :-
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true
    ; format(user_error, "~n[cpp is_list build output]~n~w~n", [OutStr]),
      throw(cpp_is_list_build_failed(Status))
    ).

run_case(ExePath, Pred, Arg, Expect) :-
    atom_string(Arg, ArgStr),
    process_create(ExePath, [Pred, ArgStr],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, _Status),
    ( Expect == true,  sub_string(OutStr, _, _, _, "true")  -> true
    ; Expect == false, sub_string(OutStr, _, _, _, "false") -> true
    ; format(user_error, "~n[cpp is_list] ~w(~w): expected ~w, got: ~w~n",
             [Pred, Arg, Expect, OutStr]),
      throw(cpp_is_list_case_failed(Pred, Arg))
    ).

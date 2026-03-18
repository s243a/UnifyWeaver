#!/usr/bin/env swipl
%% validate_targets.pl — Generate and test output for all targets with multifile dispatch
%%
%% Usage: swipl -g "consult('scripts/validate_targets'), run_validation" -t halt

:- use_module('src/unifyweaver/core/advanced/advanced_recursive_compiler').
:- use_module('src/unifyweaver/core/advanced/linear_recursion').
:- use_module('src/unifyweaver/core/advanced/tail_recursion').
:- use_module('src/unifyweaver/core/advanced/tree_recursion').
:- use_module('src/unifyweaver/core/advanced/mutual_recursion').
:- use_module('src/unifyweaver/core/advanced/multicall_linear_recursion').
:- use_module('src/unifyweaver/core/advanced/direct_multi_call_recursion').
:- use_module('src/unifyweaver/core/recursive_compiler').

% Load target modules (registers multifile dispatch clauses)
:- use_module('src/unifyweaver/targets/ruby_target', []).
:- use_module('src/unifyweaver/targets/perl_target', []).
:- use_module('src/unifyweaver/targets/typescript_target', []).
:- use_module('src/unifyweaver/targets/c_target', []).
:- use_module('src/unifyweaver/targets/cpp_target', []).
:- use_module('src/unifyweaver/targets/elixir_target', []).
:- use_module('src/unifyweaver/targets/fsharp_target', []).
:- use_module('src/unifyweaver/targets/haskell_target', []).
:- use_module('src/unifyweaver/targets/java_target', []).
:- use_module('src/unifyweaver/targets/lua_target', []).
:- use_module('src/unifyweaver/targets/r_target', []).
:- use_module('src/unifyweaver/targets/kotlin_target', []).
:- use_module('src/unifyweaver/targets/scala_target', []).
:- use_module('src/unifyweaver/targets/clojure_target', []).
:- use_module('src/unifyweaver/targets/jython_target', []).
:- use_module('src/unifyweaver/targets/rust_target', []).
:- use_module('src/unifyweaver/targets/go_target', []).
:- use_module('src/unifyweaver/targets/powershell_target', []).

:- use_module(library(lists)).

%% Setup test predicates
setup_predicates :-
    % factorial/2 — linear recursion, numeric, fold = N * F1
    catch(abolish(factorial/2), _, true),
    assertz(user:(factorial(0, 1))),
    assertz(user:(factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),

    % list_length/2 — linear recursion, list, fold = L1 + 1
    catch(abolish(list_length/2), _, true),
    assertz(user:(list_length([], 0))),
    assertz(user:(list_length([_H|T], L) :- list_length(T, L1), L is L1 + 1)),

    % list_sum/2 — linear recursion, list, fold = S1 + H (tests head var extraction)
    catch(abolish(list_sum/2), _, true),
    assertz(user:(list_sum([], 0))),
    assertz(user:(list_sum([H|T], S) :- list_sum(T, S1), S is S1 + H)),

    % fib/2 — multicall linear recursion
    catch(abolish(fib/2), _, true),
    assertz(user:(fib(0, 0))),
    assertz(user:(fib(1, 1))),
    assertz(user:(fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, fib(N1, F1), fib(N2, F2), F is F1 + F2)),

    % is_even/1, is_odd/1 — mutual recursion
    catch(abolish(is_even/1), _, true),
    catch(abolish(is_odd/1), _, true),
    assertz(user:(is_even(0))),
    assertz(user:(is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz(user:(is_odd(1))),
    assertz(user:(is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),

    % count/3 — tail recursion with accumulator
    catch(abolish(count/3), _, true),
    assertz(user:(count([], Acc, Acc))),
    assertz(user:(count([_|T], Acc, N) :- Acc1 is Acc + 1, count(T, Acc1, N))).

%% Write code to file
write_file(Path, Code) :-
    open(Path, write, Stream),
    write(Stream, Code),
    close(Stream).

%% Generate for a single target + pattern combination
generate(Target, Pattern, Pred, Arity, Filename) :-
    Dir = 'output/validation',
    atomic_list_concat([Dir, '/', Filename], FilePath),
    (   Pattern = linear ->
        compile_advanced_recursive(Pred/Arity, [target(Target)], Code)
    ;   Pattern = multicall ->
        % Force multicall via forbid_linear_recursion
        catch(abolish(forbid_linear_recursion/1), _, true),
        assertz(forbid_linear_recursion(Pred/Arity)),
        compile_advanced_recursive(Pred/Arity, [target(Target)], Code),
        catch(abolish(forbid_linear_recursion/1), _, true)
    ;   Pattern = mutual ->
        compile_advanced_recursive(is_even/1, [target(Target)], Code)
    ;   Pattern = tail ->
        compile_advanced_recursive(Pred/Arity, [target(Target)], Code)
    ;   Pattern = transitive ->
        recursive_compiler:compile_transitive_closure(Target, Pred, Arity, parent, [], Code)
    ;   fail
    ),
    write_file(FilePath, Code),
    format('  ✓ ~w -> ~w~n', [Pattern, FilePath]).

%% Run all validations
run_validation :-
    writeln(''),
    writeln('╔════════════════════════════════════════════════════════╗'),
    writeln('║  TARGET VALIDATION — Generate & Test                   ║'),
    writeln('╚════════════════════════════════════════════════════════╝'),
    writeln(''),

    % Create output directory
    (exists_directory('output/validation') -> true ; make_directory('output/validation')),

    % Setup test predicates
    setup_predicates,

    % Also define parent facts for transitive closure test
    catch(abolish(parent/2), _, true),
    assertz(user:parent(tom, bob)),
    assertz(user:parent(bob, ann)),
    assertz(user:parent(bob, pat)),
    assertz(user:parent(ann, sue)),

    % Full-suite targets (all 6 patterns: linear, list_sum, tail, multicall, mutual, transitive)
    FullTargets = [ruby, perl, typescript, c, cpp, elixir, fsharp, haskell, java, lua, r,
                   kotlin, scala, clojure, jython, rust, go],
    forall(
        member(Target, FullTargets),
        validate_target(Target)
    ),

    % Transitive-closure-only targets
    TcOnlyTargets = [sql, powershell],
    forall(
        member(Target, TcOnlyTargets),
        validate_tc_only(Target)
    ),

    writeln(''),
    writeln('╔════════════════════════════════════════════════════════╗'),
    writeln('║  GENERATION COMPLETE — Ready for runtime testing       ║'),
    writeln('╚════════════════════════════════════════════════════════╝').

validate_target(Target) :-
    target_ext(Target, Ext),
    format('~n--- ~w ---~n', [Target]),

    % Linear recursion: factorial
    atomic_list_concat(['factorial', Ext], FacFile),
    (   catch(generate(Target, linear, factorial, 2, FacFile), E,
            (format('  ✗ linear factorial: ~w~n', [E]), true))
    ->  true ; format('  ✗ linear factorial: FAILED~n')
    ),

    % Linear recursion: list_sum (tests head var aggregation)
    atomic_list_concat(['list_sum', Ext], SumFile),
    (   catch(generate(Target, linear, list_sum, 2, SumFile), E2,
            (format('  ✗ linear list_sum: ~w~n', [E2]), true))
    ->  true ; format('  ✗ linear list_sum: FAILED~n')
    ),

    % Tail recursion: count (list accumulator)
    atomic_list_concat(['count', Ext], CountFile),
    (   catch(generate(Target, tail, count, 3, CountFile), E_tail,
            (format('  ✗ tail count: ~w~n', [E_tail]), true))
    ->  true ; format('  ✗ tail count: FAILED~n')
    ),

    % Multicall: fib
    atomic_list_concat(['fib', Ext], FibFile),
    (   catch(generate(Target, multicall, fib, 2, FibFile), E3,
            (format('  ✗ multicall fib: ~w~n', [E3]), true))
    ->  true ; format('  ✗ multicall fib: FAILED~n')
    ),

    % Mutual recursion: is_even/is_odd
    atomic_list_concat(['even_odd', Ext], MutFile),
    (   catch(generate(Target, mutual, is_even, 1, MutFile), E4,
            (format('  ✗ mutual even_odd: ~w~n', [E4]), true))
    ->  true ; format('  ✗ mutual even_odd: FAILED~n')
    ),

    % Transitive closure: ancestor
    atomic_list_concat(['ancestor', Ext], TcFile),
    (   catch(generate(Target, transitive, ancestor, 2, TcFile), E5,
            (format('  ✗ transitive ancestor: ~w~n', [E5]), true))
    ->  true ; format('  ✗ transitive ancestor: not supported~n')
    ).

%% Validate transitive-closure-only targets (SQL)
validate_tc_only(Target) :-
    target_ext(Target, Ext),
    format('~n--- ~w (transitive closure only) ---~n', [Target]),
    atomic_list_concat(['ancestor', Ext], TcFile),
    (   catch(generate(Target, transitive, ancestor, 2, TcFile), E,
            (format('  ✗ transitive ancestor: ~w~n', [E]), true))
    ->  true ; format('  ✗ transitive ancestor: FAILED~n')
    ).

target_ext(ruby, '.rb').
target_ext(perl, '.pl').
target_ext(typescript, '.ts').
target_ext(c, '.c').
target_ext(cpp, '.cpp').
target_ext(elixir, '.ex').
target_ext(fsharp, '.fs').
target_ext(haskell, '.hs').
target_ext(java, '.java').
target_ext(lua, '.lua').
target_ext(r, '.R').
target_ext(go, '.go').
target_ext(sql, '.sql').
target_ext(kotlin, '.kt').
target_ext(scala, '.scala').
target_ext(clojure, '.clj').
target_ext(jython, '.jy.py').
target_ext(rust, '.rs').
target_ext(powershell, '.ps1').

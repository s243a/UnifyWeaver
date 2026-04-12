:- encoding(utf8).
% test_wam_llvm_td3_wiring.pl
% Verifies M5.5: the transitive_distance3 dispatcher stub delegates
% to @wam_td3_kernel_impl, and a default weak stub is in place so the
% pure WAM path still parses and verifies.
%
% Why not test an in-module override? LLVM disallows having both a
% weak default and a strong definition of the same function in the
% same module — the override mechanism relies on link-time resolution
% across object files. The compile pipeline (in a follow-up PR) will
% instead substitute the real body in place of the weak default at
% template-rendering time rather than appending a duplicate symbol.
%
% Checks:
%   1. @wam_td3_kernel_impl has a weak default definition in the
%      generated module.
%   2. stub_transitive_distance3 calls @wam_td3_kernel_impl.
%   3. The pure WAM module parses with llvm-as.
%   4. The pure WAM module passes opt -passes=verify.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

test_impl_weak_default :-
    format('--- @wam_td3_kernel_impl has weak default in template ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('td3w_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    % M5.8: the weak default now takes an i32 %instance parameter
    % that the dispatcher passes through from the call_foreign
    % instruction's op2.
    ( sub_string(Src, _, _, _, 'define weak i1 @wam_td3_kernel_impl(%WamState* %vm, i32 %instance)')
    -> format('  PASS: weak default present with instance parameter~n')
    ;  format('  FAIL: weak default missing~n'),
       throw(weak_default_missing)
    ),
    catch(delete_file(LLPath), _, true).

test_stub_calls_impl :-
    format('--- stub_transitive_distance3 delegates to impl ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('td3w_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    % M5.8: the stub passes the instance discriminator through so
    % the concrete impl can dispatch to the right per-predicate
    % edge table.
    ( sub_string(Src, _, _, _, 'call i1 @wam_td3_kernel_impl(%WamState* %vm, i32 %instance)')
    -> format('  PASS: stub calls @wam_td3_kernel_impl with instance~n')
    ;  format('  FAIL: stub does not delegate to impl~n'),
       throw(stub_not_wired)
    ),
    catch(delete_file(LLPath), _, true).

test_pure_wam_parses :-
    format('--- llvm-as accepts pure-WAM module ---~n'),
    ( process_which('llvm-as')
    -> test_pure_wam_llvm_as
    ;  format('  SKIP: llvm-as not found~n')
    ).

test_pure_wam_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('td3w_pure')], LLPath),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: pure-WAM module parses~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    ( process_which('opt')
    -> format(atom(VCmd),
        'opt -passes=verify -disable-output ~w 2>&1', [BCPath]),
       shell(VCmd, VExit),
       ( VExit == 0
       -> format('  PASS: opt -passes=verify accepted bitcode~n')
       ;  format('  FAIL: opt -passes=verify exit=~w~n', [VExit])
       )
    ;  format('  SKIP: opt not found~n')
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

test_all :-
    test_impl_weak_default,
    test_stub_calls_impl,
    catch(test_pure_wam_parses, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).

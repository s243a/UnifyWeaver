% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../../src/unifyweaver/targets/wam_target').

:- initialization(main, main).

meta_target(Prefix, Value, pair(Prefix, Value)).

meta_call_atom(Value, Out) :-
    call(meta_target, atom, Value, Out).

meta_call_compound(Value, Out) :-
    call(meta_target(compound), Value, Out).

main :-
    write_wam_llvm_project(
        [meta_call_atom/2, meta_call_compound/2, meta_target/3],
        [module_name(plawk_meta_call_probe)],
        'examples/plawk/generated/plawk_meta_call_probe.ll'
    ),
    halt.

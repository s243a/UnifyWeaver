% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../../src/unifyweaver/targets/wam_target').
:- use_module('../core/plawk_core').

:- initialization(main, main).

main :-
    write_wam_llvm_project(
        [plawk_core:process_all/4],
        [module_name(plawk_loop_probe)],
        'examples/plawk/generated/plawk_loop_probe.ll'
    ),
    halt.

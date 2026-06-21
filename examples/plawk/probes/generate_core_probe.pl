% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../../src/unifyweaver/targets/wam_target').
:- use_module('../core/plawk_core').

:- initialization(main, main).

main :-
    Predicates = [
        plawk_core:state_counter/2,
        plawk_core:increment_counter/2,
        plawk_core:item_field_count/2,
        plawk_core:nr/2,
        plawk_core:nf/2,
        plawk_core:fs/2,
        plawk_core:ofs/2,
        plawk_core:append_output/3,
        plawk_core:state_outputs/2,
        plawk_core:print_item/3,
        plawk_core:print_fields/3,
        plawk_core:item_field/3
    ],
    write_wam_llvm_project(
        Predicates,
        [module_name(plawk_core_probe)],
        'examples/plawk/generated/plawk_core_probe.ll'
    ),
    halt.

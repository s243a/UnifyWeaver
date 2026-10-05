% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk's .stache templates (examples/plawk/codegen/llvm/plawk_stache.pl): the
% assoc_elem pilot spells every array-element print / write line from one
% structural case per shape. Byte identity with the format/2 strings it replaced
% is checked across the whole IR corpus by the PR (ir_corpus dump + diff); this
% suite pins the renderer's contract.

:- use_module(library(plunit)).
:- use_module('../examples/plawk/codegen/llvm/plawk_stache').

:- begin_tests(plawk_stache).

test(preflight_renders_every_case) :-
    plawk_stache_preflight.

% One case serves all three printers: the kind is a value, not a case.
test(one_print_case_for_every_kind) :-
    forall(member(K, [i64, f64, str]),
           ( plawk_render_stache(assoc_elem, elem_print(4, '%key', K), L),
             format(atom(Want),
                 '  call void @wam_assoc_~w_print(%WamAssocI64Table* %plawk_assoc_table_4, i64 %key)',
                 [K]),
             assertion(L == Want) )).

% The write case per encoding carries the runtime function as a value; only the
% LLVM type word differs between the two encodings.
test(write_cases_per_encoding) :-
    plawk_render_stache(assoc_elem, elem_write('%c', 1, '%k', 1, inc, i64), L1),
    assertion(L1 == '  %c = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_1, i64 %k, i64 1)'),
    plawk_render_stache(assoc_elem, elem_write('%s', 0, 9, '1.0', add, f64), L2),
    assertion(L2 == '  %s = call double @wam_assoc_f64_add(%WamAssocI64Table* %plawk_assoc_table_0, i64 9, double 1.0)').

test(nonground_op_is_an_error, [error(instantiation_error, _)]) :-
    plawk_render_stache(assoc_elem, elem_print(_, '%k', i64), _).

% No case matches (and the template has no default): the rendering is empty,
% which the one-line policy rejects rather than emitting nothing.
test(unmatched_op_is_an_error, [error(plawk_stache(not_one_line(assoc_elem, elem_bogus(1), _)))]) :-
    plawk_render_stache(assoc_elem, elem_bogus(1), _).

:- end_tests(plawk_stache).

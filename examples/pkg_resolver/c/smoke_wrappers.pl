:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% smoke_wrappers.pl -- 1-arity smoke queries over the frozen resolver.
% Catalogs and request lists are copied from
% examples/pkg_resolver/test_resolver.pl (scenario_catalog/2 around
% lines 40-82 and corpus_case/4 around 391-398). Selection is unbound;
% the driver prints the grounded answer. Nothing here constrains the
% output to an expected term.

% corpus_case(empty_requests, empty, resolve, []).
smoke_empty_requests(Sel) :-
    resolve(catalog([], [], [], [], [], []), [], Sel).

% corpus_case(single_package, single, resolve, [foo]).
smoke_single_package(Sel) :-
    resolve(catalog([package(foo, v(1, 0, 0))], [], [], [], [], []),
            [foo], Sel).

% corpus_case(backtrack_conflict_deeper, backtrack_conflict, resolve, [a, b]).
smoke_backtrack_conflict_deeper(Sel) :-
    resolve(catalog(
                [package(a, v(2, 0, 0)), package(a, v(1, 0, 0)),
                 package(b, v(1, 0, 0)), package(c, v(1, 0, 0))],
                [depends(a, v(2, 0, 0), c, any),
                 depends(a, v(1, 0, 0), c, any)],
                [conflicts(a, v(2, 0, 0), b)],
                [], [], []),
            [a, b], Sel).

% corpus_case(unsatisfiable_missing, missing, resolve, [foo]).
smoke_unsatisfiable_missing(Sel) :-
    resolve(catalog([package(foo, v(1, 0, 0))],
                    [depends(foo, v(1, 0, 0), nosuch, any)],
                    [], [], [], []),
            [foo], Sel).

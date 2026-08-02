:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% pe_registry_mirror.pl - GENERATED closed-constraint mirror of the
% sealed v0.4 registry surface (process_cards.py REGISTRY).
%
% DO NOT EDIT: regenerate with gen_registry_mirror.py (ruling 5(b),
% DESIGN_prolog_elaborator.md / PR #4093).  The embedded sha256 of
% process_cards.py is re-checked at load; a mismatch means the sealed
% registry changed since generation, and the mirror REFUSES TO LOAD
% until regenerated (fail closed - no stale registry facts).

:- module(pe_registry_mirror, [
    pe_registry_version/1,
    pe_atom/1,
    pe_operator/1,
    pe_variadic/1,
    pe_output/2,
    pe_modifier/2,
    pe_kwspec/4,
    pe_required/2
]).

:- use_module(library(sha)).

pe_registry_version('v0.4').

pe_mirror_source_sha256('358a4b575746786551218c0127ab5bfb0905a4c0d42c7f639e19100c59ea97ae').

% pe_atom(Name): forms that may appear bare.
pe_atom(e5).
pe_atom(fs).
pe_atom(gemini).
pe_atom('gpt-5.5-low').
pe_atom(graph).
pe_atom(haiku).
pe_atom(human).
pe_atom(llm).
pe_atom(luna).
pe_atom(opus).
pe_atom(pearltrees).
pe_atom(simplemind).
pe_atom(simplewiki).
pe_atom(sonnet).

% pe_operator(Name): anything that may be applied to arguments.
pe_operator(blend).
pe_operator(distill).
pe_operator(e5).
pe_operator(hop_decay).
pe_operator(kalman).
pe_operator(lca_frac).
pe_operator(lineage).
pe_operator(margin).
pe_operator(max).
pe_operator(menu).
pe_operator(pick).
pe_operator(product).
pe_operator(routing).

% pe_variadic(Name): open-ended positional arity.
pe_variadic(blend).
pe_variadic(max).
pe_variadic(product).

% pe_output(Name, OutputType).
pe_output(blend, 'target-set').
pe_output(distill, 'target-set').
pe_output(e5, score).
pe_output(fs, substrate).
pe_output(gemini, judge).
pe_output('gpt-5.5-low', judge).
pe_output(graph, judge).
pe_output(haiku, judge).
pe_output(hop_decay, score).
pe_output(human, judge).
pe_output(kalman, 'target-set').
pe_output(lca_frac, score).
pe_output(lineage, 'target-set').
pe_output(llm, judge).
pe_output(luna, judge).
pe_output(margin, score).
pe_output(max, score).
pe_output(menu, 'target-set').
pe_output(opus, judge).
pe_output(pearltrees, substrate).
pe_output(pick, pick).
pe_output(product, score).
pe_output(routing, pick).
pe_output(simplemind, substrate).
pe_output(simplewiki, substrate).
pe_output(sonnet, judge).

% pe_modifier(Name, Modifier).
pe_modifier(graph, discrim).
pe_modifier(llm, element).
pe_modifier(llm, subcat).
pe_modifier(luna, 'D').
pe_modifier(luna, 'S').
pe_modifier(sonnet, lineage).

% pe_kwspec(Op, Kw, Kind, Default).  Kind is the registry's declared
% kind verbatim (value kinds: number/int/string/number_list/int_list/
% estimand/impl; an output-type kind such as judge marks a node-valued
% kwarg).  Default is none or default(Value).
pe_kwspec(blend, estimand, estimand, none).
pe_kwspec(blend, impl, impl, none).
pe_kwspec(blend, w, number_list, none).
pe_kwspec(distill, estimand, estimand, none).
pe_kwspec(distill, impl, impl, none).
pe_kwspec(e5, estimand, estimand, none).
pe_kwspec(e5, impl, impl, none).
pe_kwspec(hop_decay, estimand, estimand, none).
pe_kwspec(hop_decay, gamma, number, none).
pe_kwspec(hop_decay, impl, impl, none).
pe_kwspec(kalman, estimand, estimand, none).
pe_kwspec(kalman, impl, impl, none).
pe_kwspec(lca_frac, estimand, estimand, none).
pe_kwspec(lca_frac, impl, impl, none).
pe_kwspec(lineage, decay, number, default(0.85)).
pe_kwspec(lineage, depth, int, none).
pe_kwspec(lineage, estimand, estimand, none).
pe_kwspec(lineage, impl, impl, none).
pe_kwspec(lineage, mu, judge, none).
pe_kwspec(margin, estimand, estimand, none).
pe_kwspec(margin, impl, impl, none).
pe_kwspec(margin, t, number, none).
pe_kwspec(max, estimand, estimand, none).
pe_kwspec(max, impl, impl, none).
pe_kwspec(menu, n, int, none).
pe_kwspec(product, estimand, estimand, none).
pe_kwspec(product, impl, impl, none).
pe_kwspec(routing, estimand, estimand, none).
pe_kwspec(routing, impl, impl, none).
pe_kwspec(routing, manifest, string, none).
pe_kwspec(routing, menus, int_list, none).
pe_kwspec(routing, t, number_list, none).

% pe_required(Op, Kw): kwargs the registry marks required.
pe_required(hop_decay, gamma).
pe_required(margin, t).
pe_required(menu, n).

% Load-time drift check: fail closed if the sealed source moved.
pe_mirror_verify :-
    module_property(pe_registry_mirror, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/../process_cards.py'], Src),
    (   exists_file(Src)
    ->  true
    ;   throw(error(pe_registry_mirror(source_missing(Src)), _))
    ),
    read_file_to_string(Src, S, [encoding(octet)]),
    sha_hash(S, H, [algorithm(sha256), encoding(octet)]),
    hash_atom(H, Actual),
    pe_mirror_source_sha256(Expected),
    (   Actual == Expected
    ->  true
    ;   throw(error(pe_registry_mirror(source_drift(expected(Expected), actual(Actual))), _))
    ).

:- initialization(pe_mirror_verify).

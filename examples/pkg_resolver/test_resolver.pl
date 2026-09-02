:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_resolver.pl -- uw-resolve P0 + P0.5 contract corpus.
%
%   swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl

:- module(test_resolver, [
    test_resolver/0,
    scenario_catalog/2,
    corpus_case/4
]).

:- use_module(library(plunit)).
:- use_module(resolver).

test_resolver :-
    run_tests(pkg_resolver),
    format("pkg_resolver contract corpus: tests finished~n", []).

v1(v(1, 0, 0)).
v2(v(2, 0, 0)).
v11(v(1, 1, 0)).

empty_cat(catalog([], [], [], [], [], [])).

% ---------------------------------------------------------------------------
% Named catalogs used by the corpus (and by the wamjs / differential dump)
% ---------------------------------------------------------------------------

scenario_catalog(empty, Cat) :-
    empty_cat(Cat).

scenario_catalog(single, Cat) :-
    v1(V),
    Cat = catalog([package(foo, V)], [], [], [], [], []).

scenario_catalog(linear, Cat) :-
    v1(V),
    Cat = catalog(
        [package(a, V), package(b, V), package(c, V)],
        [depends(a, V, b, any), depends(b, V, c, any)],
        [], [], [], []).

scenario_catalog(backtrack_conflict, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(a, V2), package(a, V1), package(b, V1), package(c, V1)],
        [depends(a, V2, c, any), depends(a, V1, c, any)],
        [conflicts(a, V2, b)],
        [], [], []).

scenario_catalog(diamond, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(a, V1), package(b, V1), package(c, V1),
         package(d, V2), package(d, V1)],
        [depends(a, V1, b, any), depends(a, V1, c, any),
         depends(b, V1, d, gte(v(1, 0, 0))),
         depends(c, V1, d, lt(v(2, 0, 0)))],
        [], [], [], []).

scenario_catalog(conflict_pair, Cat) :-
    v1(V),
    Cat = catalog(
        [package(foo, V), package(bar, V)],
        [],
        [conflicts(foo, V, bar)],
        [], [], []).

scenario_catalog(missing, Cat) :-
    v1(V),
    Cat = catalog([package(foo, V)], [depends(foo, V, nosuch, any)], [], [], [], []).

scenario_catalog(upgradeable_base, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(lib, V1), package(lib, V2), package(app, V1)],
        [depends(app, V1, lib, any)],
        [],
        [lib-V1],
        [], []).

scenario_catalog(blocked_base, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(lib, V1), package(lib, V2), package(app, V1)],
        [depends(app, V1, lib, gte(V2))],
        [],
        [lib-V1],
        [], []).

scenario_catalog(layer_tree, Cat) :-
    v1(V),
    Cat = catalog(
        [package(a, V), package(b, V), package(c, V), package(d, V)],
        [depends(a, V, b, any), depends(a, V, d, any), depends(b, V, c, any)],
        [], [], [], []).

scenario_catalog(removal_basic, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(lib, V), package(tool, V)],
        [depends(app, V, lib, any), depends(app, V, tool, any)],
        [],
        [],
        [app-V, lib-V, tool-V],
        [app]).

scenario_catalog(removal_saved, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(other, V), package(lib, V), package(tool, V)],
        [depends(app, V, lib, any), depends(app, V, tool, any),
         depends(other, V, lib, any)],
        [],
        [],
        [app-V, other-V, lib-V, tool-V],
        [app, other]).

scenario_catalog(removal_base, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(lib, V)],
        [depends(app, V, lib, any)],
        [],
        [lib-V],
        [app-V, lib-V],
        [app]).

scenario_catalog(base_request, Cat) :-
    v1(V),
    Cat = catalog(
        [package(lib, V)],
        [],
        [],
        [lib-V],
        [lib-V],
        []).

scenario_catalog(eq_and_range, Cat) :-
    v1(V1), v11(V11), v2(V2),
    Cat = catalog(
        [package(p, V1), package(p, V11), package(p, V2), package(q, V1)],
        [depends(q, V1, p, range(V11, V2))],
        [], [], [], []).

scenario_catalog(multi_request, Cat) :-
    v1(V),
    Cat = catalog(
        [package(x, V), package(y, V), package(z, V)],
        [depends(x, V, z, any)],
        [], [], [], []).

scenario_catalog(two_ceilings, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(app, V1), package(lib, V1), package(lib, V2),
         package(old, V1), package(old, V2)],
        [depends(app, V1, lib, gte(V2)), depends(app, V1, old, gte(V2))],
        [],
        [lib-V1, old-V1],
        [], []).

% --- P0.5 catalogs (P0 catalogs above are unchanged) ---

scenario_catalog(reasons_safe, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(foot, V1), package(foot, V2),
         package(blank, V1), package(blank, V2),
         package(shadow, V1), package(shadow, V2)],
        [], [],
        [base(foot-V1, footprint), base(blank-V1, blanket),
         base(shadow-V1, layer_shadow)],
        [], []).

scenario_catalog(reasons_modified, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(patched, V1), package(patched, V2)],
        [], [],
        [base(patched-V1, modified)],
        [], []).

scenario_catalog(abi_coord, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(lib, V1), package(lib, V2),
         package(app, V1), package(app, V2),
         package(tool, V1)],
        [depends(app, V1, lib, lt(V2)),
         depends(app, V2, lib, gte(V2)),
         depends(tool, V1, lib, any)],
        [],
        [base(lib-V1, abi_anchor), app-V1, tool-V1],
        [], []).

scenario_catalog(abi_blocked, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(lib, V1), package(lib, V2),
         package(app, V1)],
        [depends(app, V1, lib, eq(V1))],
        [],
        [base(lib-V1, abi_anchor), app-V1],
        [], []).

scenario_catalog(audit_mix, Cat) :-
    v1(V1), v2(V2),
    Cat = catalog(
        [package(solo, V1), package(lib, V1), package(lib, V2),
         package(app, V1), package(patched, V1)],
        [depends(app, V1, lib, eq(V1))],
        [],
        [solo-V1, lib-V1, base(app-V1, footprint), base(patched-V1, modified)],
        [], []).

scenario_catalog(named_devx, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(lib, V), package(gcc, V)],
        [depends(app, V, lib, any), depends(lib, V, gcc, any)],
        [],
        [lib-V],
        [], [],
        [layer(devx, [gcc-V])],
        [], []).

scenario_catalog(excluded_select, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(bad, V)],
        [depends(app, V, bad, any)],
        [], [], [], [],
        [],
        [bad],
        []).

scenario_catalog(excluded_removal, Cat) :-
    v1(V),
    Cat = catalog(
        [package(app, V), package(lib, V)],
        [depends(app, V, lib, any)],
        [],
        [],
        [app-V, lib-V],
        [app],
        [],
        [lib],
        []).

scenario_catalog(what_needs, Cat) :-
    v1(V), v2(V2),
    Cat = catalog(
        [package(lib, V), package(app, V), package(app, V2),
         package(other, V), package(ghost, V)],
        [depends(app, V, lib, any), depends(app, V2, lib, gte(V)),
         depends(other, V, lib, any), depends(ghost, V, nosuch, any)],
        [],
        [lib-V],
        [app-V, lib-V],
        [app]).

scenario_catalog(alias_rxvt, Cat) :-
    v1(V),
    Cat = catalog(
        [package(rxvt, V), package(app, V)],
        [depends(app, V, rxvt, any)],
        [], [], [], [],
        [],
        [],
        [alias(urxvt, rxvt)]).

% corpus_case(Id, CatalogName, Query, Args)
% Query = resolve | resolve_layered | explain_blocked | layer_closure | removal_orphans
% Args  = requests list, a single request, or a package name.
corpus_case(empty_requests, empty, resolve, []).
corpus_case(empty_layered, empty, resolve_layered, []).
corpus_case(single_package, single, resolve, [foo]).
corpus_case(linear_closure, linear, resolve, [a]).
corpus_case(backtrack_conflict_deeper, backtrack_conflict, resolve, [a, b]).
corpus_case(diamond_one_version, diamond, resolve, [a]).
corpus_case(conflict_exclusion, conflict_pair, resolve, [foo, bar]).
corpus_case(unsatisfiable_missing, missing, resolve, [foo]).
corpus_case(range_and_eq_range, eq_and_range, resolve, [q]).
corpus_case(range_and_eq_eq, eq_and_range, resolve, [req(p, eq(v(1, 0, 0)))]).
corpus_case(multi_request, multi_request, resolve, [x, y]).
corpus_case(resolve_upgrades_base, upgradeable_base, resolve, [app]).
corpus_case(layered_base_not_reselected, upgradeable_base, resolve_layered, [app]).
corpus_case(layered_blocked_explanation, blocked_base, explain_blocked, app).
corpus_case(two_blocked_ceilings, two_ceilings, explain_blocked, app).
corpus_case(layer_manifest_dep_order, layer_tree, layer_closure, a).
corpus_case(layer_omits_base, upgradeable_base, layer_closure, app).
corpus_case(removal_orphans_basic, removal_basic, removal_orphans, app).
corpus_case(removal_orphan_saved, removal_saved, removal_orphans, app).
corpus_case(base_never_orphaned, removal_base, removal_orphans, app).
corpus_case(layered_request_already_base, base_request, resolve_layered, [lib]).
corpus_case(layered_request_already_base_layer, base_request, layer_closure, lib).
corpus_case(removal_missing_pkg, empty, removal_orphans, ghost).
% P0.5
corpus_case(safe_upgrade_footprint, reasons_safe, safe_upgrade, [foot, v(2, 0, 0)]).
corpus_case(safe_upgrade_blanket, reasons_safe, safe_upgrade, [blank, v(2, 0, 0)]).
corpus_case(safe_upgrade_shadow, reasons_safe, safe_upgrade, [shadow, v(2, 0, 0)]).
corpus_case(safe_upgrade_modified, reasons_modified, safe_upgrade, [patched, v(2, 0, 0)]).
corpus_case(safe_upgrade_no_candidate, reasons_safe, safe_upgrade, [foot, v(9, 0, 0)]).
corpus_case(safe_upgrade_coordinated, abi_coord, safe_upgrade, [lib, v(2, 0, 0)]).
corpus_case(upgrade_set_minimal, abi_coord, upgrade_set, [lib, v(2, 0, 0)]).
corpus_case(upgrade_set_blocked, abi_blocked, upgrade_set, [lib, v(2, 0, 0)]).
corpus_case(freeze_audit_mix, audit_mix, freeze_audit, []).
corpus_case(named_layer_satisfies, named_devx, resolve_layered, [app]).
corpus_case(named_layer_closure, named_devx, layer_closure, app).
corpus_case(excluded_never_selected, excluded_select, resolve, [app]).
corpus_case(excluded_does_not_block_removal, excluded_removal, removal_orphans, app).
corpus_case(dependents_lib, what_needs, dependents, lib).
corpus_case(dependents_installed_lib, what_needs, dependents_installed, lib).
corpus_case(alias_request_edge, alias_rxvt, resolve, [urxvt]).

:- begin_tests(pkg_resolver).

% 1. empty / trivial
test(empty_requests) :-
    scenario_catalog(empty, Cat),
    resolve(Cat, [], Sel),
    assertion(Sel == []).

test(empty_layered) :-
    scenario_catalog(empty, Cat),
    resolve_layered(Cat, [], Sel),
    assertion(Sel == []).

% 2. single package, no deps
test(single_package) :-
    scenario_catalog(single, Cat), v1(V),
    resolve(Cat, [foo], Sel),
    assertion(Sel == [foo-V]).

% 3. linear a→b→c
test(linear_closure) :-
    scenario_catalog(linear, Cat), v1(V),
    resolve(Cat, [a], Sel),
    assertion(Sel == [a-V, b-V, c-V]).

% 4. genuine backtracking: a-2.0 conflicts with b deeper; a-1.0 succeeds
test(backtrack_conflict_deeper) :-
    scenario_catalog(backtrack_conflict, Cat), v1(V1),
    resolve(Cat, [a, b], Sel),
    assertion(Sel == [a-V1, b-V1, c-V1]),
    \+ member(a-v(2, 0, 0), Sel).

% 5. diamond: one version of d (1.0; 2.0 fails c's lt(2.0))
test(diamond_one_version) :-
    scenario_catalog(diamond, Cat), v1(V1),
    resolve(Cat, [a], Sel),
    assertion(Sel == [a-V1, b-V1, c-V1, d-V1]),
    assertion(\+ member(d-v(2, 0, 0), Sel)).

% 6. conflict exclusion
test(conflict_exclusion) :-
    scenario_catalog(conflict_pair, Cat),
    \+ resolve(Cat, [foo, bar], _).

% 7. missing dependency
test(unsatisfiable_missing) :-
    scenario_catalog(missing, Cat),
    \+ resolve(Cat, [foo], _).

% 8. range constraint picks the middle version; eq pins
test(range_and_eq) :-
    scenario_catalog(eq_and_range, Cat), v1(V1), v11(V11),
    resolve(Cat, [q], Sel),
    assertion(member(p-V11, Sel)),
    assertion(\+ member(p-v(2, 0, 0), Sel)),
    resolve(Cat, [req(p, eq(V1))], SelEq),
    assertion(SelEq == [p-V1]).

% 9. two independent requests
test(multi_request) :-
    scenario_catalog(multi_request, Cat), v1(V),
    resolve(Cat, [x, y], Sel),
    assertion(Sel == [x-V, y-V, z-V]).

% 10. naive resolve upgrades a base package (highest lib)
test(resolve_upgrades_base) :-
    scenario_catalog(upgradeable_base, Cat), v1(V1), v2(V2),
    resolve(Cat, [app], Sel),
    assertion(member(app-V1, Sel)),
    assertion(member(lib-V2, Sel)),
    assertion(\+ member(lib-V1, Sel)).

% 11. layered uses the base lib, does not re-select it
test(layered_base_not_reselected) :-
    scenario_catalog(upgradeable_base, Cat), v1(V1),
    resolve_layered(Cat, [app], Sel),
    assertion(Sel == [app-V1]),
    assertion(\+ member(lib-_, Sel)).

% 12. layered succeeds (no upgrade) where classic resolve would upgrade
test(layered_succeeds_without_upgrade) :-
    scenario_catalog(upgradeable_base, Cat), v1(V1), v2(V2),
    resolve(Cat, [app], Classic),
    resolve_layered(Cat, [app], Layered),
    assertion(member(lib-V2, Classic)),
    assertion(Layered == [app-V1]).

% 13. unsatisfiable layered + exact blocked/3 explanation
test(layered_blocked_explanation) :-
    scenario_catalog(blocked_base, Cat), v1(V1), v2(V2),
    \+ resolve_layered(Cat, [app], _),
    findall(B, explain_blocked(Cat, app, B), Bs0),
    sort(Bs0, Bs),
    assertion(Bs == [blocked(lib, needs(gte(V2)), base_has(V1))]).

% 14. two held ceilings in one request
test(two_blocked_ceilings) :-
    scenario_catalog(two_ceilings, Cat), v1(V1), v2(V2),
    \+ resolve_layered(Cat, [app], _),
    findall(B, explain_blocked(Cat, app, B), Bs0),
    sort(Bs0, Bs),
    assertion(Bs == [
        blocked(lib, needs(gte(V2)), base_has(V1)),
        blocked(old, needs(gte(V2)), base_has(V1))
    ]).

% 15. layer manifest: dependencies before dependents (c, b, d, a)
test(layer_manifest_dep_order) :-
    scenario_catalog(layer_tree, Cat), v1(V),
    layer_closure(Cat, a, Layer),
    assertion(Layer == [c-V, b-V, d-V, a-V]).

% 16. layered layer does not include a base dependency
test(layer_omits_base) :-
    scenario_catalog(upgradeable_base, Cat), v1(V1),
    layer_closure(Cat, app, Layer),
    assertion(Layer == [app-V1]).

% 17. removal orphans both deps
test(removal_orphans_basic) :-
    scenario_catalog(removal_basic, Cat), v1(V),
    removal_orphans(Cat, app, Orphans),
    assertion(Orphans == [lib-V, tool-V]).

% 18. a would-be orphan is saved by another installed requested package
test(removal_orphan_saved) :-
    scenario_catalog(removal_saved, Cat), v1(V),
    removal_orphans(Cat, app, Orphans),
    assertion(Orphans == [tool-V]),
    assertion(\+ member(lib-_, Orphans)).

% 19. base packages are never orphans
test(base_never_orphaned) :-
    scenario_catalog(removal_base, Cat),
    removal_orphans(Cat, app, Orphans),
    assertion(Orphans == []).

% 20. requesting a package that is already the base → empty selection
test(layered_request_already_base) :-
    scenario_catalog(base_request, Cat),
    resolve_layered(Cat, [lib], Sel),
    assertion(Sel == []),
    layer_closure(Cat, lib, Layer),
    assertion(Layer == []).

% 21. removal of something not installed is empty, not an error
test(removal_missing_pkg) :-
    scenario_catalog(empty, Cat),
    removal_orphans(Cat, ghost, Orphans),
    assertion(Orphans == []).

% ---- P0.5 ---------------------------------------------------------------

test(safe_upgrade_footprint) :-
    scenario_catalog(reasons_safe, Cat), v2(V2),
    safe_upgrade(Cat, foot, V2, Verdict),
    assertion(Verdict == safe(cost(footprint))).

test(safe_upgrade_blanket) :-
    scenario_catalog(reasons_safe, Cat), v2(V2),
    safe_upgrade(Cat, blank, V2, Verdict),
    assertion(Verdict == safe(cost(blanket))).

test(safe_upgrade_layer_shadow) :-
    scenario_catalog(reasons_safe, Cat), v2(V2),
    safe_upgrade(Cat, shadow, V2, Verdict),
    assertion(Verdict == safe(cost(layer_shadow))).

test(safe_upgrade_modified) :-
    scenario_catalog(reasons_modified, Cat), v2(V2),
    safe_upgrade(Cat, patched, V2, Verdict),
    assertion(Verdict == unsafe(modified)).

test(safe_upgrade_no_candidate) :-
    scenario_catalog(reasons_safe, Cat),
    safe_upgrade(Cat, foot, v(9, 0, 0), Verdict),
    assertion(Verdict == no_candidate).

test(safe_upgrade_coordinated) :-
    scenario_catalog(abi_coord, Cat), v2(V2),
    safe_upgrade(Cat, lib, V2, Verdict),
    assertion(Verdict == coordinated([app-V2, lib-V2])).

test(upgrade_set_minimality) :-
    scenario_catalog(abi_coord, Cat), v1(V1), v2(V2),
    upgrade_set(Cat, lib, V2, Set),
    assertion(Set == [app-V2, lib-V2]),
    assertion(\+ member(tool-_, Set)),
    assertion(\+ member(app-V1, Set)).

test(upgrade_set_blocked_explanation) :-
    scenario_catalog(abi_blocked, Cat), v1(V1), v2(V2),
    \+ upgrade_set(Cat, lib, V2, _),
    upgrade_set_result(Cat, lib, V2, R),
    assertion(R == blocked(app, needs(eq(V1)), base_has(V1))).

test(freeze_audit_over_frozen_and_suggest) :-
    scenario_catalog(audit_mix, Cat),
    freeze_audit(Cat, Audit),
    assertion(member(audit(solo, over_frozen), Audit)),
    assertion(member(audit(lib, suggest(abi_anchor)), Audit)),
    assertion(member(audit(app, held(footprint)), Audit)),
    assertion(member(audit(patched, held(modified)), Audit)).

test(named_layer_satisfies_dep) :-
    scenario_catalog(named_devx, Cat), v1(V),
    resolve_layered(Cat, [app], Sel),
    assertion(Sel == [app-V]),
    assertion(\+ member(gcc-_, Sel)),
    assertion(\+ member(lib-_, Sel)).

test(named_layer_closure_omits_loaded) :-
    scenario_catalog(named_devx, Cat), v1(V),
    layer_closure(Cat, app, Layer),
    assertion(Layer == [app-V]).

test(excluded_never_selected) :-
    scenario_catalog(excluded_select, Cat),
    \+ resolve(Cat, [app], _),
    \+ resolve(Cat, [bad], _).

test(excluded_does_not_block_removal) :-
    scenario_catalog(excluded_removal, Cat), v1(V),
    removal_orphans(Cat, app, Orphans),
    assertion(Orphans == [lib-V]).

test(dependents_direct) :-
    scenario_catalog(what_needs, Cat), v1(V), v2(V2),
    dependents(Cat, lib, Ds),
    assertion(Ds == [app-V, app-V2, other-V]).

test(dependents_installed_restricted) :-
    scenario_catalog(what_needs, Cat), v1(V),
    dependents_installed(Cat, lib, Ds),
    assertion(Ds == [app-V]),
    assertion(\+ member(other-_, Ds)),
    assertion(\+ member(app-v(2, 0, 0), Ds)).

test(alias_at_request_edge) :-
    scenario_catalog(alias_rxvt, Cat), v1(V),
    resolve(Cat, [urxvt], Sel),
    assertion(Sel == [rxvt-V]),
    assertion(\+ member(urxvt-_, Sel)).

:- end_tests(pkg_resolver).

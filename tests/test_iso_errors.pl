% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_iso_errors.pl - Unit tests for the shared ISO errors module
% at src/unifyweaver/core/iso_errors.pl.
%
% These tests verify the helpers in isolation -- no target dependency.
% Per-target wrappers (rewrite_text, audit predicates) have their own
% target-specific test files.

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/core/iso_errors').

:- begin_tests(iso_errors).

iso_errors_temp_config_file(Path, Lines) :-
    tmp_file('tmp_shared_iso_cfg', Path),
    setup_call_cleanup(
        open(Path, write, Out),
        forall(member(L, Lines), format(Out, '~w~n', [L])),
        close(Out)).

%% ----- valid_pi and pi_matches -----

test(valid_pi_bare) :-
    assertion(iso_errors_valid_pi(foo/2)),
    assertion(iso_errors_valid_pi(foo/0)),
    assertion(\+ iso_errors_valid_pi(foo/(-1))),
    assertion(\+ iso_errors_valid_pi(foo)),
    assertion(\+ iso_errors_valid_pi(123/2)).

test(valid_pi_module_qualified) :-
    assertion(iso_errors_valid_pi(mymod:foo/2)),
    assertion(\+ iso_errors_valid_pi(mymod:foo)),
    assertion(\+ iso_errors_valid_pi(mymod:123/2)).

test(pi_matches_same) :-
    assertion(iso_errors_pi_matches(foo/2, foo/2)),
    assertion(iso_errors_pi_matches(mymod:foo/2, mymod:foo/2)).

test(pi_matches_bare_to_qualified) :-
    %% Bare Name/Arity in overrides matches Module:Name/Arity in any
    %% module.  This is the common case for the multi-module warning.
    assertion(iso_errors_pi_matches(foo/2, mymod:foo/2)),
    assertion(iso_errors_pi_matches(foo/2, othermod:foo/2)).

test(pi_matches_qualified_to_bare) :-
    %% Symmetric: a Module:Name/Arity override matches a bare PI in
    %% that module.  Used when the predicate list contains bare PIs.
    assertion(iso_errors_pi_matches(mymod:foo/2, foo/2)).

test(pi_matches_negative_cases) :-
    assertion(\+ iso_errors_pi_matches(foo/2, bar/2)),
    assertion(\+ iso_errors_pi_matches(foo/2, foo/3)),
    assertion(\+ iso_errors_pi_matches(mymod:foo/2, othermod:foo/2)).

%% ----- Config-file loader -----

test(load_config_basic) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(true).',
        'iso_errors_override(legacy_lookup/3, false).',
        'iso_errors_override(experimental:my_pred/2, true).'
    ]),
    setup_call_cleanup(
        true,
        (   iso_errors_load_config(Path, Config),
            assertion(Config == iso_config(true,
                [legacy_lookup/3-false,
                 (experimental:my_pred/2)-true]))
        ),
        delete_file(Path)).

test(load_config_missing_returns_defaults) :-
    %% Non-existent file -> iso_config(false, []) (silent failure
    %% per spec; lets callers run unconfigured without erroring).
    iso_errors_load_config('/tmp/no_such_iso_config_file_zzz.pl',
                           iso_config(Default, Overrides)),
    assertion(Default == false),
    assertion(Overrides == []).

test(load_config_ignores_unknown_facts) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(true).',
        'iso_errors_override(foo/0, false).',
        'unrelated_fact(blah).',
        'iso_errors_default(bad_value).'  % bad mode value -> ignored
    ]),
    setup_call_cleanup(
        true,
        (   iso_errors_load_config(Path, iso_config(Default, Overrides)),
            assertion(Default == true),
            assertion(Overrides == [foo/0-false])
        ),
        delete_file(Path)).

%% ----- Option resolution and inline-wins -----

test(resolve_options_inline_only) :-
    iso_errors_resolve_options(
        [iso_errors(true), iso_errors(foo/0, false)],
        iso_config(Default, Overrides)),
    assertion(Default == true),
    assertion(Overrides == [foo/0-false]).

test(resolve_options_inline_wins_over_file) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(false).',
        'iso_errors_override(legacy_lookup/3, false).'
    ]),
    setup_call_cleanup(
        true,
        (   iso_errors_resolve_options(
                [iso_errors_config(Path),
                 iso_errors(true),
                 iso_errors(legacy_lookup/3, true)],
                Config),
            iso_errors_mode_for(Config, user:legacy_lookup/3, M1),
            assertion(M1 == true),
            iso_errors_mode_for(Config, user:never_listed/2, M2),
            assertion(M2 == true)
        ),
        delete_file(Path)).

%% ----- Mode resolution -----

test(mode_for_bare_override_matches_qualified_pi) :-
    Config = iso_config(true, [legacy/2-false]),
    iso_errors_mode_for(Config, user:legacy/2, M1),
    assertion(M1 == false),
    iso_errors_mode_for(Config, other_module:legacy/2, M2),
    assertion(M2 == false),
    iso_errors_mode_for(Config, user:unrelated/3, M3),
    assertion(M3 == true).

test(mode_for_qualified_override_module_scoped) :-
    Config = iso_config(true, [(mymod:scoped/1)-false]),
    iso_errors_mode_for(Config, mymod:scoped/1, M1),
    assertion(M1 == false),
    %% Should NOT match a different module per pi_matches semantics.
    iso_errors_mode_for(Config, othermod:scoped/1, M2),
    assertion(M2 == true).

%% ----- Multi-module warning (smoke) -----

test(warn_multi_module_does_not_fail_on_clean_input) :-
    Config = iso_config(true, []),
    iso_errors_warn_multi_module(Config, [user:foo/0, mymod:bar/1]),
    %% Just verifying no exception/failure.
    true.

%% ----- Item-level rewrite (uses multifile tables) -----

test(rewrite_item_uses_iso_table_in_iso_mode) :-
    %% Assert a test entry, run rewrite, then retract.
    setup_call_cleanup(
        ( assertz(iso_errors:iso_errors_default_to_iso("testkey/1", "testkey_iso/1")),
          assertz(iso_errors:iso_errors_default_to_lax("testkey/1", "testkey_lax/1")) ),
        (   iso_errors_rewrite(iso_config(true, []), demo/0,
                               [builtin_call("testkey/1", 1)],
                               Out1),
            assertion(Out1 == [builtin_call("testkey_iso/1", 1)]),
            iso_errors_rewrite(iso_config(false, []), demo/0,
                               [builtin_call("testkey/1", 1)],
                               Out2),
            assertion(Out2 == [builtin_call("testkey_lax/1", 1)])
        ),
        ( retract(iso_errors:iso_errors_default_to_iso("testkey/1", "testkey_iso/1")),
          retract(iso_errors:iso_errors_default_to_lax("testkey/1", "testkey_lax/1")) )).

test(rewrite_item_passes_through_unknown_keys) :-
    %% Keys not in the tables survive unchanged.
    iso_errors_rewrite(iso_config(true, []), demo/0,
                       [builtin_call("not_in_tables/2", 2)],
                       Out),
    assertion(Out == [builtin_call("not_in_tables/2", 2)]).

%% ----- Audit walker primitives -----

test(audit_normalise_pi) :-
    iso_errors_audit_normalise_pi(foo/2, P1),
    assertion(P1 == foo/2),
    iso_errors_audit_normalise_pi(foo/2-stuff, P2),
    assertion(P2 == foo/2),
    iso_errors_audit_normalise_pi(mymod:foo/2, P3),
    assertion(P3 == mymod:foo/2).

test(audit_walk_advances_pc_for_other_items) :-
    iso_errors_audit_walk([other, other, other], 0, true, [], Out),
    assertion(Out == []).

test(audit_walk_records_builtin_call_sites) :-
    setup_call_cleanup(
        assertz(iso_errors:iso_errors_default_to_iso("foo/0", "foo_iso/0")),
        (   iso_errors_audit_walk([builtin_call("foo/0", 0)], 0, true, [], OutRev),
            reverse(OutRev, Out),
            assertion(Out == [site(0, "foo/0", "foo_iso/0", default, true)])
        ),
        retract(iso_errors:iso_errors_default_to_iso("foo/0", "foo_iso/0"))).

%% ----- key_has_suffix -----

test(key_has_suffix) :-
    assertion(iso_errors_key_has_suffix("foo_iso/2", "_iso")),
    assertion(iso_errors_key_has_suffix("foo_lax/2", "_lax")),
    assertion(\+ iso_errors_key_has_suffix("foo/2", "_iso")),
    assertion(\+ iso_errors_key_has_suffix("foo/2", "_lax")),
    %% Verify the suffix check looks at the Name portion before the
    %% slash, not the whole string.
    assertion(\+ iso_errors_key_has_suffix("foo/0", "_0")).

:- end_tests(iso_errors).

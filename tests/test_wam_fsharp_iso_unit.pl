% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_iso_unit.pl - Prolog-only unit tests for F# WAM ISO
% error config loader, rewrite, and audit predicates.
%
% These tests don''t invoke dotnet -- they exercise the
% iso_errors_resolve_options / iso_errors_mode_for /
% iso_errors_rewrite_text / wam_fsharp_iso_audit helpers directly.
% End-to-end behaviour (catch/throw + is_iso/is_lax in generated F#)
% is covered by tests/core/test_wam_fsharp_iso_smoke.pl.

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex), [delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').

:- begin_tests(wam_fsharp_iso_errors_config).

% Helper: write a temp config file with one line per Line atom.
iso_errors_temp_config_file(Path, Lines) :-
    tmp_file('tmp_wam_fsharp_iso_cfg', Path),
    setup_call_cleanup(
        open(Path, write, Out),
        forall(member(L, Lines), format(Out, '~w~n', [L])),
        close(Out)).

test(iso_errors_config_loader_basic) :-
    iso_errors_temp_config_file(Path, [
        'iso_errors_default(true).',
        'iso_errors_override(legacy_lookup/3, false).',
        'iso_errors_override(experimental:my_pred/2, true).',
        'ignored_fact(ok).'
    ]),
    setup_call_cleanup(
        true,
        (   iso_errors_load_config(Path, Config),
            assertion(Config == iso_config(true,
                [legacy_lookup/3-false,
                 (experimental:my_pred/2)-true])),
            iso_errors_mode_for(Config, user:legacy_lookup/3, M1),
            assertion(M1 == false),
            iso_errors_mode_for(Config, user:never_listed/2, M2),
            assertion(M2 == true),
            iso_errors_mode_for(Config, experimental:my_pred/2, M3),
            assertion(M3 == true)
        ),
        delete_file(Path)).

test(iso_errors_config_missing_default) :-
    iso_errors_temp_config_file(Path, [
        '% no default declared',
        'iso_errors_override(foo/0, true).'
    ]),
    setup_call_cleanup(
        true,
        (   iso_errors_load_config(Path, Config),
            % Absent default -> false per spec.
            assertion(Config == iso_config(false, [foo/0-true])),
            iso_errors_mode_for(Config, user:foo/0, M1),
            assertion(M1 == true),
            iso_errors_mode_for(Config, user:bar/0, M2),
            assertion(M2 == false)
        ),
        delete_file(Path)).

test(iso_errors_inline_wins_over_file) :-
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

test(iso_errors_text_rewrite_is_to_is_iso) :-
    Wam0 = 'demo/0:\n  put_variable 4, 1\n  put_integer 1, 2\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call is_iso/2 2")),
    \+ sub_string(IsoWam, _, _, _, "builtin_call is/2 2").

test(iso_errors_text_rewrite_is_to_is_lax) :-
    Wam0 = 'demo/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call is_lax/2 2")),
    \+ sub_string(LaxWam, _, _, _, "builtin_call is/2 2").

test(iso_errors_text_rewrite_explicit_iso_survives) :-
    % Explicit is_iso/2 in source must NOT be rewritten in either mode.
    Wam0 = 'demo/0:\n  builtin_call is_iso/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call is_iso/2 2")),
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    % Lax mode shouldn''t touch the explicit iso form either.
    assertion(sub_string(LaxWam, _, _, _, "builtin_call is_iso/2 2")).

test(iso_errors_text_rewrite_comparison_iso) :-
    % Arithmetic-compare sweep: each of the 6 ops rewrites to its
    % _iso variant under iso_config(true, []).
    Wam0 = 'cmp/0:\n  builtin_call </2 2\n  builtin_call >/2 2\n  builtin_call >=/2 2\n  builtin_call =</2 2\n  builtin_call =:=/2 2\n  builtin_call =\\=/2 2',
    iso_errors_rewrite_text(iso_config(true, []), cmp/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call <_iso/2 2")),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call >_iso/2 2")),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call >=_iso/2 2")),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call =<_iso/2 2")),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call =:=_iso/2 2")),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call =\\=_iso/2 2")).

test(iso_errors_text_rewrite_comparison_lax) :-
    Wam0 = 'cmp/0:\n  builtin_call </2 2\n  builtin_call >/2 2\n  builtin_call =:=/2 2',
    iso_errors_rewrite_text(iso_config(false, []), cmp/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call <_lax/2 2")),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call >_lax/2 2")),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call =:=_lax/2 2")).

test(iso_errors_text_rewrite_succ) :-
    Wam0 = 'succ_demo/0:\n  builtin_call succ/2 2',
    iso_errors_rewrite_text(iso_config(true, []), succ_demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call succ_iso/2 2")),
    iso_errors_rewrite_text(iso_config(false, []), succ_demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call succ_lax/2 2")).

test(iso_errors_text_rewrite_per_pred_override) :-
    % Default ISO, override demo/0 to lax.
    Config = iso_config(true, [demo/0-false]),
    Wam0 = 'demo/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(Config, demo/0, Wam0, OutWam),
    assertion(sub_string(OutWam, _, _, _, "builtin_call is_lax/2 2")),
    % Other predicates default to ISO.
    Wam2 = 'other/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(Config, other/0, Wam2, OutWam2),
    assertion(sub_string(OutWam2, _, _, _, "builtin_call is_iso/2 2")).

test(iso_errors_audit_structure) :-
    setup_call_cleanup(
        assertz((user:fs_iso_audit_pred :- X is 1 + 2, X = 3)),
        (   wam_fsharp_iso_audit(
                [user:fs_iso_audit_pred/0],
                [iso_errors(fs_iso_audit_pred/0, true)],
                Audit),
            assertion(Audit = [audit(user:fs_iso_audit_pred/0, true, _Sites)])
        ),
        retractall(user:fs_iso_audit_pred)).

test(iso_errors_project_generation_rewrites_wam_text) :-
    TmpDir = '/tmp/uw_fs_iso_rewrite_unit',
    catch(delete_directory_and_contents(TmpDir), _, true),
    setup_call_cleanup(
        assertz((user:fs_iso_rewrite_demo :- X is 1 + 2, X == 3)),
        (   write_wam_fsharp_project(
                [user:fs_iso_rewrite_demo/0],
                [iso_errors(true), no_kernels(true)],
                TmpDir),
            % Predicates.fs should contain BuiltinCall ("is_iso/2", ...)
            % rather than BuiltinCall ("is/2", ...).
            string_concat(TmpDir, '/Predicates.fs', PredsPath),
            read_file_to_string(PredsPath, Code, []),
            assertion(sub_string(Code, _, _, _, "is_iso/2")),
            \+ sub_string(Code, _, _, _, "is/2,")
        ),
        once((   retractall(user:fs_iso_rewrite_demo),
                 catch(delete_directory_and_contents(TmpDir), _, true)
             ))).

:- end_tests(wam_fsharp_iso_errors_config).

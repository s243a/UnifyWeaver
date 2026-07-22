% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_r_iso_unit.pl - Prolog-only unit tests for R WAM ISO-R-0
% (shared config/rewrite/audit + is/2 key table). End-to-end generated-R
% behaviour is covered by tests/test_wam_r_iso_smoke.pl.

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex), [delete_directory_and_contents/1]).
:- use_module('../src/unifyweaver/targets/wam_r_target').

:- begin_tests(wam_r_iso_errors_config).

iso_errors_temp_config_file(Path, Lines) :-
    tmp_file('tmp_wam_r_iso_cfg', Path),
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

test(iso_errors_module_scoped_and_bare_pi) :-
    Config = iso_config(false, [
        (mod_a:arith/1)-true,
        bare_arith/1-true
    ]),
    iso_errors_mode_for(Config, mod_a:arith/1, M1),
    assertion(M1 == true),
    iso_errors_mode_for(Config, mod_b:arith/1, M2),
    assertion(M2 == false),
    iso_errors_mode_for(Config, user:bare_arith/1, M3),
    assertion(M3 == true).

test(iso_errors_text_rewrite_builtin_call_is_to_iso) :-
    Wam0 = 'demo/0:\n  put_variable 4, 1\n  put_integer 1, 2\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call is_iso/2 2")),
    \+ sub_string(IsoWam, _, _, _, "builtin_call is/2 2").

test(iso_errors_text_rewrite_builtin_call_is_to_lax) :-
    Wam0 = 'demo/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call is_lax/2 2")),
    \+ sub_string(LaxWam, _, _, _, "builtin_call is/2 2").

test(iso_errors_text_rewrite_put_structure_call_execute) :-
    Wam0 = 'demo/0:\n  put_structure is/2, A1\n  call is/2 2\n  execute is/2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "put_structure is_iso/2")),
    assertion(sub_string(IsoWam, _, _, _, "call is_iso/2")),
    assertion(sub_string(IsoWam, _, _, _, "execute is_iso/2")),
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "put_structure is_lax/2")),
    assertion(sub_string(LaxWam, _, _, _, "call is_lax/2")),
    assertion(sub_string(LaxWam, _, _, _, "execute is_lax/2")).

test(iso_errors_text_rewrite_explicit_iso_survives) :-
    Wam0 = 'demo/0:\n  builtin_call is_iso/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call is_iso/2 2")),
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call is_iso/2 2")).

test(iso_errors_text_rewrite_explicit_lax_survives) :-
    Wam0 = 'demo/0:\n  builtin_call is_lax/2 2\n  proceed',
    iso_errors_rewrite_text(iso_config(true, []), demo/0, Wam0, IsoWam),
    assertion(sub_string(IsoWam, _, _, _, "builtin_call is_lax/2 2")),
    iso_errors_rewrite_text(iso_config(false, []), demo/0, Wam0, LaxWam),
    assertion(sub_string(LaxWam, _, _, _, "builtin_call is_lax/2 2")).

test(iso_errors_text_rewrite_per_pred_override) :-
    Config = iso_config(true, [demo/0-false]),
    Wam0 = 'demo/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(Config, demo/0, Wam0, OutWam),
    assertion(sub_string(OutWam, _, _, _, "builtin_call is_lax/2 2")),
    Wam2 = 'other/0:\n  builtin_call is/2 2\n  proceed',
    iso_errors_rewrite_text(Config, other/0, Wam2, OutWam2),
    assertion(sub_string(OutWam2, _, _, _, "builtin_call is_iso/2 2")).

test(iso_errors_audit_structure) :-
    setup_call_cleanup(
        assertz((user:r_iso_audit_pred :- X is 1 + 2, X = 3)),
        (   wam_r_iso_audit(
                [user:r_iso_audit_pred/0],
                [iso_errors(r_iso_audit_pred/0, true)],
                Audit),
            assertion((
                Audit = [audit(user:r_iso_audit_pred/0, true, Sites)],
                Sites \= [],
                memberchk(site(_, "is/2", "is_iso/2", default, true), Sites)
            ))
        ),
        retractall(user:r_iso_audit_pred)).

test(iso_errors_project_generation_selects_concrete_key) :-
    once((
        TmpDir = '/tmp/uw_r_iso_rewrite_unit',
        catch(delete_directory_and_contents(TmpDir), _, true),
        setup_call_cleanup(
            assertz((user:r_iso_rewrite_demo :- X is 1 + 2, X == 3)),
            (   write_wam_r_project(
                    [user:r_iso_rewrite_demo/0],
                    [iso_errors(true)],
                    TmpDir),
                string_concat(TmpDir, '/R/generated_program.R', ProgPath),
                read_file_to_string(ProgPath, Code, []),
                assertion(sub_string(Code, _, _, _, "is_iso/2")),
                assertion(\+ sub_string(Code, _, _, _, 'BuiltinCall("is/2"'))
            ),
            (   retractall(user:r_iso_rewrite_demo),
                catch(delete_directory_and_contents(TmpDir), _, true)
            )
        )
    )).

test(iso_errors_project_generation_preserves_module_scope) :-
    once((
        TmpDir = '/tmp/uw_r_iso_module_scope_unit',
        catch(delete_directory_and_contents(TmpDir), _, true),
        setup_call_cleanup(
            assertz((mod_b:r_iso_scope_demo :- X is 1 + 2, X == 3)),
            (   write_wam_r_project(
                    [mod_b:r_iso_scope_demo/0],
                    [iso_errors(false),
                     iso_errors(mod_a:r_iso_scope_demo/0, true)],
                    TmpDir),
                string_concat(TmpDir, '/R/generated_program.R', ProgPath),
                read_file_to_string(ProgPath, Code, []),
                assertion(sub_string(Code, _, _, _, "is_lax/2")),
                assertion(\+ sub_string(Code, _, _, _, 'BuiltinCall("is_iso/2"'))
            ),
            (   retractall(mod_b:r_iso_scope_demo),
                catch(delete_directory_and_contents(TmpDir), _, true)
            )
        )
    )).

:- end_tests(wam_r_iso_errors_config).

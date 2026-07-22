% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_r_iso_smoke.pl - ISO-R-0 generated-R smoke for is/2 three-form.
% Hand-written PASS/FAIL; exit nonzero on failure (F# smoke convention).
%
% Usage: swipl -q -f init.pl -s tests/test_wam_r_iso_smoke.pl -g main -t halt

:- encoding(utf8).
:- use_module('../src/unifyweaver/targets/wam_r_target',
              [write_wam_r_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).

pass(Name) :- format('[PASS] ~w~n', [Name]).
fail_test(Name, Reason) :-
    format('[FAIL] ~w: ~w~n', [Name, Reason]),
    nb_setval(r_iso_smoke_failed, true).

rscript_available :-
    catch(
        ( process_create(path('Rscript'), ['--version'],
                         [stdout(null), stderr(null), process(PID)]),
          process_wait(PID, exit(0))
        ), _, fail).

run_rscript(Drive, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path('Rscript'), [Drive],
                       [cwd(Dir), stdout(pipe(Out)), stderr(pipe(Err)),
                        process(PID)]),
        ( read_string(Out, _, OutStr),
          read_string(Err, _, ErrStr),
          process_wait(PID, exit(ExitCode)),
          string_concat(OutStr, ErrStr, OutText)
        ),
        ( catch(close(Out), _, true), catch(close(Err), _, true) )).

write_driver(Path, Checks) :-
    open(Path, write, S),
    format(S,
'source("R/wam_runtime.R")
source("R/generated_program.R")
passes <- 0L; fails <- 0L
check <- function(name, key) {
  start_pc <- shared_labels[[key]]
  ok <- tryCatch(
    !is.null(start_pc) && isTRUE(WamRuntime$run_predicate(shared_program, start_pc, list())),
    error = function(e) { message(conditionMessage(e)); FALSE })
  if (isTRUE(ok)) { passes <<- passes + 1L; cat(sprintf("[PASS] %s\\n", name)) }
  else { fails <<- fails + 1L; cat(sprintf("[FAIL] %s\\n", name)) }
}
', []),
    forall(member(Name-Key, Checks),
           format(S, 'check("~w", "~w")~n', [Name, Key])),
    write(S,
'cat(sprintf("RESULT %d/%d\\n", passes, passes + fails))
quit(status = if (fails == 0L) 0L else 1L)
'),
    close(S).

unique_tmp(Base, Dir) :-
    get_time(T), format(atom(Dir), '/tmp/~w_~w', [Base, T]),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

run_project(Label, Preds, Opts, Checks) :-
    format('--- ~w ---~n', [Label]),
    unique_tmp('uw_r_iso_smoke', Tmp),
    write_wam_r_project(Preds, Opts, Tmp),
    directory_file_path(Tmp, 'drive.R', Drive),
    write_driver(Drive, Checks),
    (   rscript_available
    ->  run_rscript(Drive, Tmp, Exit, Out),
        write(Out),
        (   Exit == 0 -> pass(Label)
        ;   fail_test(Label, ['Rscript exit=', Exit])
        )
    ;   directory_file_path(Tmp, 'R/generated_program.R', Prog),
        read_file_to_string(Prog, Code, []),
        (   (sub_string(Code,_,_,_,"is_iso/2"); sub_string(Code,_,_,_,"is_lax/2"))
        ->  pass(Label-codegen-skip-rscript)
        ;   fail_test(Label, 'missing rewritten is key in codegen')
        )
    ),
    catch(delete_directory_and_contents(Tmp), _, true).

main :-
    nb_setval(r_iso_smoke_failed, false),
    % Helpers + scenarios
    assertz((user:r_ok :- X is 1 + 2, X == 3)),
    assertz((user:r_iso_inst :-
        catch(is_iso(_, _Y), error(instantiation_error, _), true))),
    assertz((user:r_iso_type :-
        catch(is_iso(_, foo), error(type_error(evaluable, _), _), true))),
    assertz((user:r_iso_zdiv :-
        catch(is_iso(_, 1 / 0), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_lax_bad :- \+ is_lax(_, foo))),
    assertz((user:r_plain_zdiv :-
        catch((_X is 1 / 0), error(evaluation_error(zero_divisor), _), true))),
    assertz((user:r_plain_bad :- \+ (_ is foo))),
    assertz((user:r_plain_ok_iso :- X is 40 + 2, X == 42)),

    % 1) successful is/2 (ISO default rewrite still succeeds)
    run_project('successful is/2 (iso default, interpreter)',
        [user:r_ok/0],
        [iso_errors(true)],
        ['successful is/2'-'r_ok/0']),

    % 2) ISO-default structured errors via plain is/2 rewrite
    run_project('ISO-default structured errors (interpreter)',
        [user:r_plain_zdiv/0, user:r_plain_ok_iso/0],
        [iso_errors(true)],
        ['ISO-default zero_divisor'-'r_plain_zdiv/0',
         'ISO-default success'-'r_plain_ok_iso/0']),

    % 3) lax-default non-throwing
    run_project('lax-default silent fail (interpreter)',
        [user:r_plain_bad/0],
        [iso_errors(false)],
        ['lax-default silent'-'r_plain_bad/0']),

    % 4) per-predicate override (global ISO, one pred lax)
    run_project('per-predicate override to lax (interpreter)',
        [user:r_plain_bad/0, user:r_plain_zdiv/0],
        [iso_errors(true), iso_errors(r_plain_bad/0, false)],
        ['override lax silent'-'r_plain_bad/0',
         'sibling stays ISO'-'r_plain_zdiv/0']),

    % 5) explicit is_iso bypasses lax mode
    run_project('explicit is_iso under lax mode (interpreter)',
        [user:r_iso_type/0, user:r_iso_inst/0, user:r_iso_zdiv/0],
        [iso_errors(false)],
        ['explicit is_iso type_error'-'r_iso_type/0',
         'explicit is_iso instantiation_error'-'r_iso_inst/0',
         'explicit is_iso zero_divisor'-'r_iso_zdiv/0']),

    % 6) explicit is_lax bypasses ISO mode
    run_project('explicit is_lax under ISO mode (interpreter)',
        [user:r_lax_bad/0],
        [iso_errors(true)],
        ['explicit is_lax silent'-'r_lax_bad/0']),

    % 7) functions/lowered emit mode
    run_project('functions mode ISO-default + explicit',
        [user:r_ok/0, user:r_iso_type/0, user:r_lax_bad/0, user:r_plain_zdiv/0],
        [emit_mode(functions), iso_errors(true)],
        ['functions successful is'-'r_ok/0',
         'functions explicit is_iso'-'r_iso_type/0',
         'functions explicit is_lax'-'r_lax_bad/0',
         'functions ISO-default zdiv'-'r_plain_zdiv/0']),

    forall(member(P, [r_ok/0, r_iso_inst/0, r_iso_type/0, r_iso_zdiv/0,
                      r_lax_bad/0, r_plain_zdiv/0, r_plain_bad/0,
                      r_plain_ok_iso/0]),
           retractall(user:P)),
    (   nb_getval(r_iso_smoke_failed, true) -> halt(1) ; halt(0) ).

:- encoding(utf8).
% End-to-end execution smoke test for the WAM ILAsm (CIL) target.
%
% The target's generated output previously did not even assemble
% (.tail directive, C#-style pseudo-code in IL bodies, invalid
% .cctor_NAME identifiers, types nested in Program while referenced
% top-level, missing default ctors, receiver-after-argument operand
% order, concrete generic memberrefs, labels[PC]=PC label tables,
% placeholder call/deallocate bodies, run_loop stack imbalance, and
% missing cut_ite/jump lowerings and var/nonvar/atom/=/2 builtins).
% This test pins the repaired pipeline end to end: generate -> ilasm
% -> mono, covering the six diagnostic probes plus the M143 var-var
% alias checks (X = Y must ALIAS the cells, not copy an unbound
% marker: conflicting later bindings must fail, and a binding must be
% visible through a three-variable chain).
%
% Gated on ilasm + mono; the probe predicates are driven by a tiny
% IL driver assembly (no C# compiler needed). Modeled on
% tests/test_wam_cpp_var_alias.pl.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_ilasm_target').

:- dynamic user:ilsmoke_p1/1.
:- dynamic user:ilsmoke_p2/1.
:- dynamic user:ilsmoke_p3/1.
:- dynamic user:ilsmoke_p4/1.
:- dynamic user:ilsmoke_p5/1.
:- dynamic user:ilsmoke_p6/1.
:- dynamic user:ilsmoke_conflict/1.
:- dynamic user:ilsmoke_chain3/1.

user:ilsmoke_p1(R) :- var(X), X = done, R is 1.
user:ilsmoke_p2(R) :- var(X), var(X), R is 1.
user:ilsmoke_p3(R) :- ( var(X), nonvar(X) -> R is 0 ; R is 1 ).
user:ilsmoke_p4(R) :- var(X), atom(foo), ( var(X) -> R is 1 ; R is 0 ).
user:ilsmoke_p5(R) :- X = Y, X = 42, ( Y =:= 42 -> R is 1 ; R is 0 ).
user:ilsmoke_p6(R) :- ( 1 =:= 1 -> R is 1 ; R is 0 ).
% Conflicting bindings through a var-var alias must FAIL.
user:ilsmoke_conflict(R) :- X = Y, X = 1, Y = 2, R is 1.
% Binding must propagate through a three-deep alias chain.
user:ilsmoke_chain3(R) :- X = Y, Y = Z, X = 42, ( Z =:= 42 -> R is 1 ; R is 0 ).

ilasm_mono_available :-
    absolute_file_name(path(ilasm), _, [access(execute), file_errors(fail)]),
    absolute_file_name(path(mono), _, [access(execute), file_errors(fail)]).

:- begin_tests(wam_ilasm_exec_smoke, [condition(ilasm_mono_available)]).

test(probes_and_alias_exec) :-
    Dir = 'output/test_wam_ilasm_exec_smoke',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    cil_atom_table_reset,
    atomic_list_concat([Dir, '/ilsmoke.il'], IlFile),
    write_wam_ilasm_project(
        [user:ilsmoke_p1/1, user:ilsmoke_p2/1, user:ilsmoke_p3/1,
         user:ilsmoke_p4/1, user:ilsmoke_p5/1, user:ilsmoke_p6/1,
         user:ilsmoke_conflict/1, user:ilsmoke_chain3/1],
        [module_name(ilsmoke)], IlFile),
    % (Pred, Arg, ExpectTrue): R=1 / R=0 query pairs read back R's value.
    Cases = [ ilsmoke_p1-1-true,       ilsmoke_p1-0-false,
              ilsmoke_p2-1-true,       ilsmoke_p2-0-false,
              ilsmoke_p3-1-true,       ilsmoke_p3-0-false,
              ilsmoke_p4-1-true,       ilsmoke_p4-0-false,
              ilsmoke_p5-1-true,       ilsmoke_p5-0-false,
              ilsmoke_p6-1-true,       ilsmoke_p6-0-false,
              ilsmoke_conflict-1-false,
              ilsmoke_chain3-1-true ],
    write_driver_il(Dir, Cases),
    shell_ok(Dir, 'ilasm /dll ilsmoke.il'),
    shell_ok(Dir, 'ilasm /exe driver.il'),
    run_driver(Dir, Lines),
    findall(Expect, member(_-_-Expect, Cases), Expected),
    check_results(Cases, Expected, Lines).

:- end_tests(wam_ilasm_exec_smoke).

% Emit a minimal IL driver: one Console.WriteLine(bool) per case, in
% case order. Wrappers take (WamState vm, Value a1); vm is null (the
% wrapper builds its own state) and a1 is IntegerValue(Arg).
write_driver_il(Dir, Cases) :-
    findall(Block, (
        member(Pred-Arg-_ , Cases),
        format(atom(Block),
'    ldnull
    ldc.i8 ~w
    newobj instance void [ilsmoke]IntegerValue::.ctor(int64)
    call bool [ilsmoke]PrologGenerated.Program::~w(class [ilsmoke]WamState, class [ilsmoke]Value)
    call void [mscorlib]System.Console::WriteLine(bool)', [Arg, Pred])
    ), Blocks),
    atomic_list_concat(Blocks, '\n', Body),
    format(atom(DriverIl),
'.assembly extern mscorlib {}
.assembly extern ilsmoke {}
.assembly driver {}
.class public auto ansi Driver extends [mscorlib]System.Object {
  .method public static void Main(string[] args) cil managed {
    .entrypoint
    .maxstack 4
~w
    ret
  }
}
', [Body]),
    atomic_list_concat([Dir, '/driver.il'], Path),
    setup_call_cleanup(
        open(Path, write, S, [encoding(utf8)]),
        format(S, "~w", [DriverIl]),
        close(S)).

shell_ok(Dir, Cmd) :-
    format(atom(Full), 'cd ~w && ~w 2>&1', [Dir, Cmd]),
    process_create(path(sh), ['-c', Full],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true
    ; format(user_error, "~n[ilasm exec smoke build output]~n~w~n", [OutStr]),
      throw(ilasm_exec_smoke_build_failed(Cmd, Status))
    ).

run_driver(Dir, Lines) :-
    format(atom(Full), 'cd ~w && mono driver.exe 2>&1', [Dir]),
    process_create(path(sh), ['-c', Full],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, _),
    split_string(OutStr, "\n", " \r\t", Raw),
    exclude(==(""), Raw, Lines).

check_results(Cases, Expected, Lines) :-
    length(Expected, N),
    ( length(Lines, N) -> true
    ; format(user_error, "~n[ilasm exec smoke] expected ~w result lines, got: ~w~n",
             [N, Lines]),
      throw(ilasm_exec_smoke_bad_output(Lines))
    ),
    forall(nth0(I, Cases, Pred-Arg-Expect),
           ( nth0(I, Lines, Line),
             ( Expect == true,  Line == "True"  -> true
             ; Expect == false, Line == "False" -> true
             ; format(user_error,
                  "~n[ilasm exec smoke] ~w(~w): expected ~w, got: ~w~n",
                  [Pred, Arg, Expect, Line]),
               throw(ilasm_exec_smoke_case_failed(Pred, Arg))
             ))).

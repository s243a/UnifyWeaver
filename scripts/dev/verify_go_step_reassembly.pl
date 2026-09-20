:- encoding(utf8).
% One-off Phase 3 re-assembly check (design §7.1): prove the five library files
% + step shell reassemble byte-identically to compile_step_wam_to_go/2 BEFORE
% the adapter/target are changed. Deleted after Phase 3.

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module('../../src/unifyweaver/core/template_system', [render_template/3]).
:- use_module('../../src/unifyweaver/targets/wam_go_target',
              [compile_step_wam_to_go/2]).

order(['GetConstant','GetVariable','GetValue','GetStructure','GetList',
       'UnifyVariable','UnifyValue','UnifyConstant',
       'PutConstant','PutVariable','PutValue','PutStructure','PutList',
       'SetVariable','SetValue','SetConstant',
       'Allocate','Deallocate','Call','GetArgInto','CallForeign',
       'CallIndexedAtomFact2','CallFactStream','CallPc','Execute','ExecutePc',
       'Jump','JumpPc','CutIte','GetLevel','Cut','BeginAggregate','EndAggregate',
       'Proceed','BuiltinCall','BuiltinExecute',
       'TryMeElse','TryMeElsePc','RetryMeElse','RetryMeElsePc','TrustMe',
       'Try','Retry','Trust',
       'SwitchOnConstant','SwitchOnConstantPc','SwitchOnStructure',
       'SwitchOnStructurePc','SwitchOnConstantA2','SwitchOnConstantA2Pc']).

lib('templates/targets/go_wam/step/head_unification.go.mustache').
lib('templates/targets/go_wam/step/body_construction.go.mustache').
lib('templates/targets/go_wam/step/control.go.mustache').
lib('templates/targets/go_wam/step/choice_point.go.mustache').
lib('templates/targets/go_wam/step/indexing.go.mustache').

% Read a lib file, strip its one final LF, return the body-of-match text.
lib_text(Path, Stripped) :-
    read_file_to_string(Path, Raw, [encoding(utf8)]),
    string_concat(Stripped, "\n", Raw).

% Case name -> body, searching all lib files.
case_body(Name, Body) :-
    format(string(TagSub), "{{case ~w}}", [Name]),
    lib(Path), lib_text(Path, Text),
    sub_string(Text, _, _, _, TagSub),
    !,
    render_template(Text, [instr=Name], Rendered),
    ( string_concat("\n", Mid, Rendered), string_concat(Body, "\n", Mid), Body \== ""
    -> true ; throw(bad_shape(Name, Rendered)) ).

case_line(Name, Line) :-
    case_body(Name, Body),
    format(string(Line), "    case *~w:\n~w", [Name, Body]).

run :-
    order(Names),
    maplist(case_line, Names, Lines),
    atomic_list_concat(Lines, '\n', CasesCode),
    read_file_to_string('templates/targets/go_wam/step/step_shell.go.mustache', ShellRaw, [encoding(utf8)]),
    string_concat(Shell, "\n", ShellRaw),
    split_string(Shell, "", "", _),  % force string
    ( sub_string(Shell, At, 9, _, "{{cases}}")
    -> sub_string(Shell, 0, At, _, Before),
       End is At + 9, sub_string(Shell, End, _, 0, After),
       atomic_list_concat([Before, CasesCode, After], Assembled)
    ; throw(no_cases_marker) ),
    compile_step_wam_to_go([], Orig0),
    atom_string(Orig0, Orig),
    atom_string(Assembled, AssembledS),
    ( AssembledS == Orig
    -> string_length(AssembledS, L),
       format("REASSEMBLY BYTE-IDENTICAL (~d chars)~n", [L])
    ; string_length(AssembledS, LA), string_length(Orig, LO),
      format("MISMATCH assembled=~d orig=~d~n", [LA, LO]),
      first_diff(AssembledS, Orig, 0)
    ).

first_diff(A, B, I) :-
    ( sub_string(A, I, 1, _, CA), sub_string(B, I, 1, _, CB)
    -> ( CA == CB -> I1 is I+1, first_diff(A, B, I1)
       ; format("first diff at ~d: assembled=~q orig=~q~n", [I, CA, CB]),
         Ctx is max(0, I-40), sub_string(A, Ctx, 80, _, SA),
         format("ctx assembled: ~q~n", [SA]) )
    ; format("length diverge at ~d~n", [I]) ).

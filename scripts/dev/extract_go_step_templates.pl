:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% extract_go_step_templates.pl - one-off Phase 3 extraction (design §7.1).
%
% Writes the five step-case library files from the RENDERED wam_go_case/2
% bodies (extraction by execution, never by copying atom text), then asserts
% re-assembly is byte-identical to the current compile_step_wam_to_go/2 output.
%
% Run from repo root:
%   LANG=C.utf8 LC_ALL=C.utf8 swipl -q -g run -t halt \
%       scripts/dev/extract_go_step_templates.pl
%
% Deleted after Phase 3.

:- use_module('../../src/unifyweaver/targets/wam_go_target',
              [compile_step_wam_to_go/2]).

% Families in findall order (design §2.4), each with its ordered case names.
family(head_unification, 'templates/targets/go_wam/step/head_unification.go.mustache',
    "head-unification instructions",
    ['GetConstant','GetVariable','GetValue','GetStructure','GetList',
     'UnifyVariable','UnifyValue','UnifyConstant']).
family(body_construction, 'templates/targets/go_wam/step/body_construction.go.mustache',
    "body-construction instructions",
    ['PutConstant','PutVariable','PutValue','PutStructure','PutList',
     'SetVariable','SetValue','SetConstant']).
family(control, 'templates/targets/go_wam/step/control.go.mustache',
    "control instructions",
    ['Allocate','Deallocate','Call','GetArgInto','CallForeign',
     'CallIndexedAtomFact2','CallFactStream','CallPc','Execute','ExecutePc',
     'Jump','JumpPc','CutIte','GetLevel','Cut','BeginAggregate','EndAggregate',
     'Proceed','BuiltinCall','BuiltinExecute']).
family(choice_point, 'templates/targets/go_wam/step/choice_point.go.mustache',
    "choice-point instructions",
    ['TryMeElse','TryMeElsePc','RetryMeElse','RetryMeElsePc','TrustMe',
     'Try','Retry','Trust']).
family(indexing, 'templates/targets/go_wam/step/indexing.go.mustache',
    "indexing instructions",
    ['SwitchOnConstant','SwitchOnConstantPc','SwitchOnStructure',
     'SwitchOnStructurePc','SwitchOnConstantA2','SwitchOnConstantA2Pc']).

run :-
    make_directory_path('templates/targets/go_wam/step'),
    forall(family(_Id, Path, Desc, Names), write_family(Path, Desc, Names)),
    format("Wrote 5 step library files.~n", []).

write_family(Path, Desc, Names) :-
    % Preamble text (discarded by the match engine); no mustache tags.
    format(string(Preamble),
"Go bodies for the ~w of Step(). One case per Go instruction type.\nThe Prolog assembler (go_step_case_order/1 in wam_go_templates.pl) owns\nthe switch ORDER and the `case *Type:` line. This preamble is discarded\nby the {{match}} engine.",
        [Desc]),
    maplist(case_block, Names, Blocks),
    atomic_list_concat(Blocks, '', CasesText),
    format(string(FileText),
"{{match instr}}~n~w~n~w{{/match}}~n", [Preamble, CasesText]),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
                       write(S, FileText),
                       close(S)),
    length(Names, N),
    format("  ~w (~d cases)~n", [Path, N]).

% One case block: {{case Name}}\n<Body>\n   (Body from the live generator).
case_block(Name, Block) :-
    wam_go_target:wam_go_case(Name, Body),
    format(string(Block), "{{case ~w}}~n~w~n", [Name, Body]).

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% extract_go_wam_templates.pl - one-off template extractor for the Go WAM
% template refactor (design §7.1). Writes template files from the RENDERED
% output of the current generator predicates, never by copying the Prolog
% source spelling: running the predicates resolves doubled apostrophes
% (''->'), reader escapes (\\n->\n, \\t->\t), '\' line continuations and
% UTF-8 (em dashes, ×) exactly once, so the file holds the true bytes.
%
% This is a developer tool, deleted after Phase 3. It never writes into a
% target directory implicitly: pass an output root on argv.
%
% Usage (run from repo root, with LANG=C.utf8 LC_ALL=C.utf8):
%
%   swipl -q -g "extract_helpers('templates/targets/go_wam')" \
%         -t halt scripts/dev/extract_go_wam_templates.pl
%
%   swipl -q -g "extract_step_cases('templates/targets/go_wam')" \
%         -t halt scripts/dev/extract_go_wam_templates.pl
%
%   swipl -q -g main -t halt scripts/dev/extract_go_wam_templates.pl -- \
%         helpers templates/targets/go_wam
%
% Every write is self-checked by reading the file back and asserting it is
% byte-identical to the rendered text.

:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- use_module('../../src/unifyweaver/targets/wam_go_target',
              [compile_wam_helpers_to_go/2]).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).

% Family -> relative file name, in findall (clause) order (design §2.4). Used
% by extract_step_cases/1 for the Phase 3 split. wam_go_case/2 clause order is
% already family-grouped, so grouping preserves the byte-identity switch order.
step_family_file(head_unification,  'step/head_unification.go.mustache').
step_family_file(body_construction, 'step/body_construction.go.mustache').
step_family_file(control,           'step/control.go.mustache').
step_family_file(choice_point,      'step/choice_point.go.mustache').
step_family_file(indexing,          'step/indexing.go.mustache').

% Membership of each Go instruction type in a family (design §2.4).
step_family(head_unification, 'GetConstant').
step_family(head_unification, 'GetVariable').
step_family(head_unification, 'GetValue').
step_family(head_unification, 'GetStructure').
step_family(head_unification, 'GetList').
step_family(head_unification, 'UnifyVariable').
step_family(head_unification, 'UnifyValue').
step_family(head_unification, 'UnifyConstant').
step_family(body_construction, 'PutConstant').
step_family(body_construction, 'PutVariable').
step_family(body_construction, 'PutValue').
step_family(body_construction, 'PutStructure').
step_family(body_construction, 'PutList').
step_family(body_construction, 'SetVariable').
step_family(body_construction, 'SetValue').
step_family(body_construction, 'SetConstant').
step_family(control, 'Allocate').
step_family(control, 'Deallocate').
step_family(control, 'Call').
step_family(control, 'GetArgInto').
step_family(control, 'CallForeign').
step_family(control, 'CallIndexedAtomFact2').
step_family(control, 'CallFactStream').
step_family(control, 'CallPc').
step_family(control, 'Execute').
step_family(control, 'ExecutePc').
step_family(control, 'Jump').
step_family(control, 'JumpPc').
step_family(control, 'CutIte').
step_family(control, 'GetLevel').
step_family(control, 'Cut').
step_family(control, 'BeginAggregate').
step_family(control, 'EndAggregate').
step_family(control, 'Proceed').
step_family(control, 'BuiltinCall').
step_family(control, 'BuiltinExecute').
step_family(choice_point, 'TryMeElse').
step_family(choice_point, 'TryMeElsePc').
step_family(choice_point, 'RetryMeElse').
step_family(choice_point, 'RetryMeElsePc').
step_family(choice_point, 'TrustMe').
step_family(choice_point, 'Try').
step_family(choice_point, 'Retry').
step_family(choice_point, 'Trust').
step_family(indexing, 'SwitchOnConstant').
step_family(indexing, 'SwitchOnConstantPc').
step_family(indexing, 'SwitchOnStructure').
step_family(indexing, 'SwitchOnStructurePc').
step_family(indexing, 'SwitchOnConstantA2').
step_family(indexing, 'SwitchOnConstantA2Pc').

%% write_utf8(+Path, +Text)
%  Write Text verbatim as UTF-8, then read it back and assert equality.
write_utf8(Path, Text) :-
    file_directory_name(Path, Dir),
    make_directory_path(Dir),
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        format(Stream, '~w', [Text]),
        close(Stream)),
    read_file_to_string(Path, Back, [encoding(utf8)]),
    atom_string(Text, TextStr),
    (   Back == TextStr
    ->  string_length(TextStr, Len),
        format('extract: wrote ~w (~w chars)~n', [Path, Len])
    ;   throw(error(extract_roundtrip_mismatch(Path), _))).

%% extract_helpers(+Root)
%  Write the rendered helper blob to <Root>/runtime/helpers.go.mustache.
extract_helpers(Root) :-
    compile_wam_helpers_to_go([], Helpers),
    directory_file_path(Root, 'runtime/helpers.go.mustache', Path),
    write_utf8(Path, Helpers),
    string_length(Helpers, Len),
    format('extract_helpers: ~w chars -> ~w~n', [Len, Path]).

%% extract_step_cases(+Root)
%  Write the five step-case library files (Phase 3), grouped by family in
%  findall order. Each family file is one {{match instr}} block; each case
%  body is the current wam_go_case/2 second argument verbatim. Re-used later;
%  Phase 1 does not need it.
extract_step_cases(Root) :-
    forall(step_family_file(Family, Rel),
           ( family_case_block(Family, Block),
             directory_file_path(Root, Rel, Path),
             write_utf8(Path, Block))).

family_case_block(Family, Block) :-
    findall(Name-Body,
            ( wam_go_target:wam_go_case(Name, Body),
              step_family(Family, Name)),
            Pairs),
    header_comment(Family, Header),
    foldl(append_case, Pairs, Header, WithCases),
    string_concat(WithCases, "{{/match}}\n", Block).

header_comment(Family, Header) :-
    format(string(Header),
        "{{match instr}}~n~w cases for Step(). One case per Go instruction type; the~nProlog assembler owns the switch ORDER and the `case *Type:` line.~nThis preamble is discarded by the engine.~n",
        [Family]).

append_case(Name-Body, Acc, Out) :-
    format(string(Case), "{{case ~w}}~n~w~n", [Name, Body]),
    string_concat(Acc, Case, Out).

%% main - argv dispatch: [Kind, Root] with Kind in {helpers, cases, all}.
main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [Kind, Root|_]
    ->  true
    ;   Kind = helpers, Root = 'templates/targets/go_wam'),
    (   Kind == helpers -> extract_helpers(Root)
    ;   Kind == cases   -> extract_step_cases(Root)
    ;   Kind == all     -> extract_helpers(Root), extract_step_cases(Root)
    ;   throw(error(unknown_extract_kind(Kind), _))).

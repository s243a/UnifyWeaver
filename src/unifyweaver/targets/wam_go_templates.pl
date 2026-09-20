:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_go_templates.pl - template adapter for the Go WAM target.
%
% Mirrors src/unifyweaver/targets/wam_cpp_templates.pl. Loads the static Go
% assets that used to live as single-quoted atoms in wam_go_target.pl, splices
% them into shells on ordered exactly-once markers, and never re-renders the
% spliced Go through a template engine (so Go composite literals like
% []qd{{source, 0}} stay literal).
%
% Phase 2 scope (design docs/proposals/wam_go_template_refactor_design.md §8):
% the helper blob is split into eight ordered static sections
% (runtime/{run_loop,aggregate,foreign_registry,atom_fact2_sources,
% foreign_results,native_kernels,execute_foreign,seek_fact_source}.go.mustache),
% each spliced into its own marker in runtime.go.mustache alongside the package
% name and the (still Prolog-assembled) Step() method. Re-concatenation with the
% shell's inter-slot separator is byte-identical to the Phase 1 single-section
% render.
%
% Fail-closed: a missing/empty/malformed asset throws a contextual error and
% preflight runs BEFORE the output directory is created, so a broken template
% never leaves a half-written project.

:- module(wam_go_templates, [
    go_render_template/3,          % +Id, +Vars, -Text
    go_render_template_at_root/4,  % +Root, +Id, +Vars, -Text
    go_render_helper_methods/1,    % -HelpersText   (helper section, one final LF stripped)
    go_render_helper_methods_at_root/2,
    go_step_case_order/1,          % -[GoTypeName,...]   the 50 names, in today's findall order
    go_step_case_text/2,           % +GoTypeName, -Body  one library case, standalone-trimmed
    go_step_case_text_at_root/3,   % +Root, +GoTypeName, -Body
    go_render_step_method/1,       % -StepText           assembled Step() (compile_step_wam_to_go/2)
    go_render_step_method_at_root/2,
    go_render_lowered_function/4,  % +Name, +Comment, +Body, -[Header,Body,Footer]
    go_render_lowered_function_at_root/5,
    go_render_ite_shell/6,         % +Indent, +Cond, +Then, +FreshReset, +Else, -Text
    go_render_ite_shell_at_root/7,
    go_render_head_match/5,         % +Indent, +Comment, +Ai, +GoVal, -Text
    go_render_head_match_at_root/6,
    go_render_file/3,               % +Id, +Vars, -Text   (go.mod/value/instructions/state/main_bench)
    go_render_file_at_root/4,
    go_render_project_lib_header/4,          % +Package, +Imports, +Predicates, -Text
    go_render_project_lib_header_at_root/5,
    go_render_project_atoms_with_table/3,    % +Package, +AtomTable, -Text
    go_render_project_atoms_with_table_at_root/4,
    go_render_project_atoms_runtime_only/2,  % +Package, -Text
    go_render_project_atoms_runtime_only_at_root/3,
    go_render_project_lowered_header/3,       % +Package, +Lowered, -Text
    go_render_project_lowered_header_at_root/4,
    go_render_project_main_parallel/3,        % +Module, +Package, -Text
    go_render_project_main_parallel_at_root/4,
    go_preflight/0,                % load + validate + render every required asset once
    go_with_template_root_for_test/2
]).

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(filesex), [directory_file_path/3]).
:- use_module('../core/template_system', [render_template/3]).

:- thread_local test_go_template_root/1.

% -- Registry: template ID -> path relative to templates/targets/go_wam ------

% Shells carry ordered variable/child slots; sections are variable-free static
% Go spliced verbatim after a Go-aware tag lint.
go_template_path(runtime_shell,   'runtime.go.mustache').

% Lowered-emitter shells (design §8 Phase 4 slice 1). Rendered by their own
% entry points (go_render_lowered_function/4, go_render_ite_shell/6), NOT through
% go_render_template/3: the body/child text is spliced as a value and never
% rescanned, so a lowered fragment containing literal {{...}} (e.g. the atom
% '{{Ai}}') stays literal.
go_template_path(lowered_function, 'lowered/function.go.mustache').
go_template_path(lowered_ite,      'lowered/ite.go.mustache').
% Head-match fragment (design §8 Phase 4 slice 2): the shared get_constant/
% get_integer/get_nil bind-or-match body. Spliced on ordered markers, NEVER
% passed through render_template/3 — a Comment holding a placeholder-shaped atom
% ('{{Ai}}') would otherwise be rescanned by sequential key replacement.
go_template_path(head_match,       'lowered/head_match.go.mustache').

% Project files (design §8 Phase 4 slice 3). The five variable-render files keep
% the generic template engine (they carry {{package_name}}/{{date}}/{{module_name}}
% and no generated Go is spliced into them), read fail-closed through the adapter
% (no more silent "// Template not found"). The five atom-embedded blocks are
% spliced (never render_template) so generated Go bodies with literal {{...}} stay
% literal.
go_file_template_path(go_mod,       'go.mod.mustache').
go_file_template_path(value,        'value.go.mustache').
go_file_template_path(instructions, 'instructions.go.mustache').
go_file_template_path(state,        'state.go.mustache').
go_file_template_path(main_bench,   'main_bench.go.mustache').

go_project_path(project_lib,               'project/lib.go.mustache').
go_project_path(project_atoms_with_table,  'project/atoms_with_table.go.mustache').
go_project_path(project_atoms_runtime_only,'project/atoms_runtime_only.go.mustache').
go_project_path(project_lowered,           'project/lowered.go.mustache').
go_project_path(project_main_parallel,     'project/main_parallel.go.mustache').

% The eight ordered helper sections the runtime shell is spliced from (design
% docs/proposals/wam_go_template_refactor_design.md §2.3 / §5). Phase 2 split
% the single runtime/helpers.go.mustache blob into these at the §2.3 group
% boundaries; re-concatenation with the shell's inter-slot separator is
% byte-exact with the Phase 1 render.
go_section_path(runtime_run_loop,           'runtime/run_loop.go.mustache').
go_section_path(runtime_aggregate,          'runtime/aggregate.go.mustache').
go_section_path(runtime_foreign_registry,   'runtime/foreign_registry.go.mustache').
go_section_path(runtime_atom_fact2_sources, 'runtime/atom_fact2_sources.go.mustache').
go_section_path(runtime_foreign_results,    'runtime/foreign_results.go.mustache').
go_section_path(runtime_native_kernels,     'runtime/native_kernels.go.mustache').
go_section_path(runtime_execute_foreign,    'runtime/execute_foreign.go.mustache').
go_section_path(runtime_seek_fact_source,   'runtime/seek_fact_source.go.mustache').

% Fixed order in which the eight helper sections are spliced into the shell and
% joined by go_render_helper_methods/1. This IS the byte-identity contract for
% the helper region; moving a declaration between section files is fine as long
% as this order and each file stay coherent.
go_helper_section_order([
    runtime_run_loop,
    runtime_aggregate,
    runtime_foreign_registry,
    runtime_atom_fact2_sources,
    runtime_foreign_results,
    runtime_native_kernels,
    runtime_execute_foreign,
    runtime_seek_fact_source
]).

% Ordered marker list for the runtime shell. Exactly one of each, in this order:
% the package name, the Prolog-assembled Step() method, then the eight helper
% sections (design §5.1).
go_shell_markers(runtime_shell,
    ["{{package_name}}", "{{step_method}}",
     "{{run_loop}}", "{{aggregate}}", "{{foreign_registry}}",
     "{{atom_fact2_sources}}", "{{foreign_results}}", "{{native_kernels}}",
     "{{execute_foreign}}", "{{seek_fact_source}}"]).

% -- Step() switch: shell + five literal-match libraries (design §5.2-§5.4) ---
%
% Phase 3 moved the 50 wam_go_case/2 snippets out of wam_go_target.pl into five
% {{match instr}} library files, one per family. The Prolog order list below
% owns the switch order and the `case *Type:` line; the library files own only
% the Go bodies. A body may move between family files without changing output as
% long as go_step_case_order/1 and the case-set check (union == order) hold.
go_step_shell_path('step/step_shell.go.mustache').

go_step_library(head_unification,  'step/head_unification.go.mustache').
go_step_library(body_construction, 'step/body_construction.go.mustache').
go_step_library(control,           'step/control.go.mustache').
go_step_library(choice_point,      'step/choice_point.go.mustache').
go_step_library(indexing,          'step/indexing.go.mustache').

% The five family files, in the order they are loaded and scanned.
go_step_library_order([head_unification, body_construction, control,
                       choice_point, indexing]).

% The 50 Go instruction type names in today's findall (= clause) order. THIS is
% the byte-identity contract for the switch (design §2.4).
go_step_case_order([
    % Head Unification (8)
    'GetConstant', 'GetVariable', 'GetValue', 'GetStructure', 'GetList',
    'UnifyVariable', 'UnifyValue', 'UnifyConstant',
    % Body Construction (8)
    'PutConstant', 'PutVariable', 'PutValue', 'PutStructure', 'PutList',
    'SetVariable', 'SetValue', 'SetConstant',
    % Control (20)
    'Allocate', 'Deallocate', 'Call', 'GetArgInto', 'CallForeign',
    'CallIndexedAtomFact2', 'CallFactStream', 'CallPc', 'Execute', 'ExecutePc',
    'Jump', 'JumpPc', 'CutIte', 'GetLevel', 'Cut', 'BeginAggregate',
    'EndAggregate', 'Proceed', 'BuiltinCall', 'BuiltinExecute',
    % Choice Point (8)
    'TryMeElse', 'TryMeElsePc', 'RetryMeElse', 'RetryMeElsePc', 'TrustMe',
    'Try', 'Retry', 'Trust',
    % Indexing (6)
    'SwitchOnConstant', 'SwitchOnConstantPc', 'SwitchOnStructure',
    'SwitchOnStructurePc', 'SwitchOnConstantA2', 'SwitchOnConstantA2Pc'
]).

% -- Go-aware tag lint (design §6.4) ----------------------------------------
%
% Reject a template-origin tag that would otherwise be emitted unresolved:
% any DECLARED marker of any Go template, or any STRUCTURAL tag. Ignore
% everything else, in particular Go composite literals ({{source, 0}},
% {{next, source}}) and bare }} — those are program text the splicing renderer
% never interprets. This is deliberately NOT the C++ "no {{ at all" lint, which
% would reject valid Go.

% Every marker any Go template declares (present and near-future phases). None
% of these ever appear inside valid Go, so their presence in a static span is
% a template-authoring error.
go_reserved_marker("{{package_name}}").
go_reserved_marker("{{module_name}}").
go_reserved_marker("{{date}}").
go_reserved_marker("{{step_method}}").
go_reserved_marker("{{helper_methods}}").
go_reserved_marker("{{cases}}").
go_reserved_marker("{{run_loop}}").
go_reserved_marker("{{aggregate}}").
go_reserved_marker("{{foreign_registry}}").
go_reserved_marker("{{atom_fact2_sources}}").
go_reserved_marker("{{foreign_results}}").
go_reserved_marker("{{native_kernels}}").
go_reserved_marker("{{execute_foreign}}").
go_reserved_marker("{{seek_fact_source}}").
% Lowered-shell markers (Phase 4 slice 1). None appears inside valid Go.
go_reserved_marker("{{comment}}").
go_reserved_marker("{{name}}").
go_reserved_marker("{{body}}").
go_reserved_marker("{{indent}}").
go_reserved_marker("{{condition}}").
go_reserved_marker("{{then}}").
go_reserved_marker("{{else}}").
go_reserved_marker("{{fresh_reset}}").
% Head-match fragment markers (Phase 4 slice 2). None appears inside valid Go.
go_reserved_marker("{{I}}").
go_reserved_marker("{{Comment}}").
go_reserved_marker("{{Ai}}").
go_reserved_marker("{{GoVal}}").
% Project-block body markers (Phase 4 slice 3). None appears inside valid Go.
go_reserved_marker("{{imports}}").
go_reserved_marker("{{predicates}}").
go_reserved_marker("{{atom_table}}").
go_reserved_marker("{{lowered}}").

go_structural_tag("{{#").
go_structural_tag("{{^").
go_structural_tag("{{/").
go_structural_tag("{{>").
go_structural_tag("{{!").
go_structural_tag("{{match ").
go_structural_tag("{{case ").
go_structural_tag("{{default}}").

% A source span must carry no template-origin tag.
go_source_span_clean(Text) :-
    \+ ( go_reserved_marker(Marker), sub_string(Text, _, _, _, Marker) ),
    \+ ( go_structural_tag(Tag),     sub_string(Text, _, _, _, Tag) ).

% -- Root resolution --------------------------------------------------------
%
% Module-relative only (a thread-local override serves isolated fixtures). No
% cwd lookup and no project option can redirect it, so a stray templates/ dir
% in the working directory can never win.
go_with_template_root_for_test(Root, Goal) :-
    setup_call_cleanup(
        asserta(test_go_template_root(Root), Ref),
        call(Goal),
        erase(Ref)).

go_template_root(Root) :-
    test_go_template_root(Root),
    !.
go_template_root(Root) :-
    source_file(go_render_template(_, _, _), Source),
    file_directory_name(Source, ModuleDir),
    directory_file_path(ModuleDir, '../../../templates/targets/go_wam', Root).

% -- Asset reading ----------------------------------------------------------
%
% Every read is explicit UTF-8 (the Go bodies carry em dashes and ×), so
% template bytes never depend on the locale.
go_read_asset(Path, Id, Which, Context, Text) :-
    catch(read_file_to_string(Path, Text0, [encoding(utf8)]),
          Error,
          throw(error(go_wam_template_load(Id, Path, Error),
                      context(Context, Which)))),
    (   Text0 == ""
    ->  throw(error(go_wam_template_empty(Id, Path),
                    context(Context, Which)))
    ;   Text = Text0
    ).

% -- Sections: literal static Go, one final LF stripped (design §6.3, §7.3) --
go_section_text(Id, Text) :-
    go_template_root(Root),
    go_section_text_at_root(Root, Id, Text).

go_section_text_at_root(Root, Id, Text) :-
    (   go_section_path(Id, Relative)
    ->  true
    ;   throw(error(domain_error(go_wam_section, Id),
                    context(go_section_text/2, 'unknown Go WAM section')))
    ),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, Id, section, go_section_text/2, Raw),
    (   string_concat(Text, "\n", Raw)
    ->  true
    ;   throw(error(go_wam_template_eol(Id, Path),
                    context(go_section_text/2, Id)))
    ),
    (   go_source_span_clean(Text)
    ->  true
    ;   throw(error(go_wam_template_tags(Id, Path),
                    context(go_section_text/2, Id)))
    ).

% -- Helper methods (Phase 2: the eight ordered sections joined by the shell) -
%
% Reads the eight helper sections (each already stripped of its final LF) and
% joins them with the runtime shell's inter-slot separator ("\n\n"), so the
% result is byte-identical to the Phase 1 single-blob render. The exported
% wam_go_target:compile_wam_helpers_to_go/2 re-appends "\n" to keep its old
% trailing-LF contract; the runtime shell owns the inter-slot newlines.
go_render_helper_methods(Text) :-
    go_template_root(Root),
    go_render_helper_methods_at_root(Root, Text).

go_render_helper_methods_at_root(Root, Text) :-
    go_helper_section_order(Ids),
    maplist(go_section_text_at_root(Root), Ids, Sections),
    atomics_to_string(Sections, "\n\n", Text).

% -- Shell splicing ---------------------------------------------------------
go_render_template(Id, Vars, Text) :-
    go_template_root(Root),
    go_render_template_at_root(Root, Id, Vars, Text).

go_render_template_at_root(Root, runtime_shell, Vars, Text) :-
    !,
    (   go_valid_runtime_shell_vars(Vars, Pkg, Step)
    ->  true
    ;   throw(error(domain_error(go_wam_template_variables, Vars),
                    context(go_render_template/3, runtime_shell)))
    ),
    go_template_path(runtime_shell, Relative),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, runtime_shell, shell, go_render_template/3, Shell),
    go_shell_markers(runtime_shell, Markers),
    (   go_split_ordered_markers(Shell, Markers, Spans),
        maplist(go_source_span_clean, Spans)
    ->  go_helper_section_order(Ids),
        maplist(go_section_text_at_root(Root), Ids, Sections),
        go_text_string(Pkg, PkgText),
        go_text_string(Step, StepText),
        go_interleave_slots(Spans, [PkgText, StepText | Sections], Parts),
        atomics_to_string(Parts, "", Text)
    ;   throw(error(go_wam_template_tags(runtime_shell, Path),
                    context(go_render_template/3, runtime_shell)))
    ).
go_render_template_at_root(_Root, Id, _Vars, _Text) :-
    throw(error(domain_error(go_wam_template, Id),
                context(go_render_template/3, 'unknown Go WAM template'))).

% runtime.go's shell has two variable slots (package name, the Prolog-assembled
% Step method); the helper section is a child the adapter renders itself.
go_valid_runtime_shell_vars(Vars, Pkg, Step) :-
    ground(Vars),
    Vars = [package_name=Pkg, step_method=Step],
    go_text_value(Pkg),
    go_text_value(Step).

go_text_value(Value) :- atom(Value), !.
go_text_value(Value) :- string(Value).

go_text_string(Value, Value) :- string(Value), !.
go_text_string(Value, String) :- atom_string(Value, String).

% Split Text on each marker in order, each exactly once. Static spans between
% and around the markers are returned for linting; inserted values are never
% rescanned (splicing, not sequential substitution).
go_split_ordered_markers(Text, [], [Text]).
go_split_ordered_markers(Text, [Marker|Markers], [Before|Parts]) :-
    string_length(Marker, Length),
    findall(At, sub_string(Text, At, Length, _, Marker), [At]),
    sub_string(Text, 0, At, _, Before),
    End is At + Length,
    sub_string(Text, End, _, 0, After),
    go_split_ordered_markers(After, Markers, Parts).

go_interleave_slots([Part], [], [Part]) :- !.
go_interleave_slots([Part|Parts], [Value|Values], [Part, Value|Joined]) :-
    go_interleave_slots(Parts, Values, Joined).

% -- Step libraries: read, scan case tags, validate the case set -------------
%
% A library file legitimately contains {{match }}/{{case }} tags, so it is NOT
% run through go_source_span_clean; instead the adapter reads it, strips its one
% final LF (like a section), scans its {{case NAME}} tags, and renders each case
% body through the generic engine's {{match instr}} lookup (design §6.3).

% Read a library file and strip its required single final LF. Never linted for
% tags (it is a match library, not a static span).
go_step_library_text_at_root(Root, LibId, Text) :-
    (   go_step_library(LibId, Relative)
    ->  true
    ;   throw(error(domain_error(go_wam_step_library, LibId),
                    context(go_step_library_text/2, 'unknown Go WAM step library')))
    ),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, LibId, step_library, go_step_library_text/2, Raw),
    (   string_concat(Text, "\n", Raw)
    ->  true
    ;   throw(error(go_wam_template_eol(LibId, Path),
                    context(go_step_library_text/2, LibId)))
    ).

% All {{case NAME}} tag names in a library text, in file order.
go_step_case_names_in(Text, Names) :-
    findall(Name, go_step_case_tag(Text, Name), Names).

go_step_case_tag(Text, Name) :-
    sub_string(Text, Begin, _, _, "{{case "),
    Start is Begin + 7,
    sub_string(Text, Start, _, 0, Tail),
    once(sub_string(Tail, EndRel, 2, _, "}}")),
    sub_string(Tail, 0, EndRel, _, ValStr),
    atom_string(Name, ValStr).

% Load all five libraries in order: lib(Id, Text, CaseNames).
go_step_libraries_at_root(Root, Libs) :-
    go_step_library_order(Ids),
    maplist(go_load_step_library(Root), Ids, Libs).

go_load_step_library(Root, Id, lib(Id, Text, Names)) :-
    go_step_library_text_at_root(Root, Id, Text),
    go_step_case_names_in(Text, Names).

% The union of case names across the five files must equal go_step_case_order/1
% exactly: no duplicate, no missing, no unknown (design §5.4).
go_validate_step_case_set(Libs) :-
    findall(Name, (member(lib(_, _, Ns), Libs), member(Name, Ns)), All),
    (   append(_, [Dup|Rest], All), memberchk(Dup, Rest)
    ->  throw(error(go_wam_step_case_duplicate(Dup),
                    context(go_validate_step_case_set/1, Dup)))
    ;   true
    ),
    go_step_case_order(Order),
    (   member(Missing, Order), \+ memberchk(Missing, All)
    ->  throw(error(go_wam_step_case_missing(Missing),
                    context(go_validate_step_case_set/1, Missing)))
    ;   true
    ),
    (   member(Unknown, All), \+ memberchk(Unknown, Order)
    ->  throw(error(go_wam_step_case_unknown(Unknown),
                    context(go_validate_step_case_set/1, Unknown)))
    ;   true
    ).

% Render one case body from already-loaded libraries. The generic engine's
% {{match instr}} selects the case; the adapter passes only instr, so Go text
% (composite literals, bare }}) is never touched. The result is exactly
% "\n" ++ Body ++ "\n" (the LF after {{case}} and the LF before the next tag);
% both are stripped to recover today's wam_go_case/2 second argument.
go_step_case_body_from_libs(Libs, Name, Body) :-
    (   member(lib(_, Text, Names), Libs), memberchk(Name, Names)
    ->  true
    ;   throw(error(go_wam_step_case_missing(Name),
                    context(go_step_case_text/2, Name)))
    ),
    render_template(Text, [instr=Name], Rendered),
    (   string_concat("\n", Mid, Rendered),
        string_concat(Body, "\n", Mid),
        Body \== ""
    ->  true
    ;   throw(error(go_wam_step_case_shape(Name),
                    context(go_step_case_text/2, Name)))
    ).

% -- Public per-case accessor (design §6.1) ----------------------------------
go_step_case_text(Name, Body) :-
    go_template_root(Root),
    go_step_case_text_at_root(Root, Name, Body).

go_step_case_text_at_root(Root, Name, Body) :-
    go_step_libraries_at_root(Root, Libs),
    go_validate_step_case_set(Libs),
    go_step_case_body_from_libs(Libs, Name, Body).

% -- Step() assembly: five libraries + shell splice (design §6.2, Appendix A) --
%
% Prolog owns the order, the `case *Type:` line and the '\n' separator; the
% library files own the Go bodies; the shell owns the header/footer and the
% {{cases}} slot. Byte-identical to the old atom-assembled compile_step_wam_to_go/2.
go_render_step_method(Step) :-
    go_template_root(Root),
    go_render_step_method_at_root(Root, Step).

go_render_step_method_at_root(Root, Step) :-
    go_step_libraries_at_root(Root, Libs),
    go_validate_step_case_set(Libs),
    go_step_case_order(Names),
    maplist(go_step_case_line(Libs), Names, Lines),
    atomic_list_concat(Lines, '\n', Cases),
    go_splice_step_shell(Root, Cases, Step).

go_step_case_line(Libs, Name, Line) :-
    go_step_case_body_from_libs(Libs, Name, Body),
    format(atom(Line), '    case *~w:\n~w', [Name, Body]).

% Splice the assembled cases into the step shell's single {{cases}} marker. The
% shell's one final LF is stripped so compile_step_wam_to_go/2 still returns
% text with no trailing LF (design §7.3).
go_splice_step_shell(Root, Cases, Step) :-
    go_step_shell_path(Relative),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, step_shell, shell, go_render_step_method/1, Raw),
    (   string_concat(Shell, "\n", Raw)
    ->  true
    ;   throw(error(go_wam_template_eol(step_shell, Path),
                    context(go_render_step_method/1, step_shell)))
    ),
    (   go_split_ordered_markers(Shell, ["{{cases}}"], [Before, After]),
        go_source_span_clean(Before),
        go_source_span_clean(After)
    ->  atomics_to_string([Before, Cases, After], "", Step)
    ;   throw(error(go_wam_template_tags(step_shell, Path),
                    context(go_render_step_method/1, step_shell)))
    ).

% -- Lowered function shell (design §8 Phase 4 slice 1) ----------------------
%
% One shell for the three lowered-function header/footer format/2s (plain, T4,
% T5/T6). Returns the [Header, Body, Footer] list lower_predicate_to_go/4 wants:
% the {{body}} marker only SPLITS the footer from the header — Body is spliced as
% the middle list element and never rescanned. Byte-identical to the old
% atom-format headers (frozen by the whole-function digests, §8 slice 0).
go_render_lowered_function(Name, Comment, Body, Lines) :-
    go_template_root(Root),
    go_render_lowered_function_at_root(Root, Name, Comment, Body, Lines).

go_render_lowered_function_at_root(Root, Name, Comment, Body,
                                   [Header, BodyText, Footer]) :-
    go_template_path(lowered_function, Relative),
    directory_file_path(Root, Relative, Path),
    (   go_nonempty_text(Name), go_nonempty_text(Comment), go_text_value(Body)
    ->  true
    ;   throw(error(domain_error(go_wam_lowered_function_variables,
                                 [Name, Comment, Body]),
                    context(go_render_lowered_function/4, Path)))
    ),
    go_read_asset(Path, lowered_function, shell, go_render_lowered_function/4,
                  Template),
    (   go_split_ordered_markers(Template,
            ["{{comment}}", "{{name}}", "{{body}}"],
            [BeforeComment, BetweenCN, BetweenNB, Footer]),
        maplist(go_source_span_clean,
                [BeforeComment, BetweenCN, BetweenNB, Footer])
    ->  go_text_string(Comment, CommentText),
        go_text_string(Name, NameText),
        go_text_string(Body, BodyText),
        atomics_to_string([BeforeComment, CommentText, BetweenCN, NameText,
                           BetweenNB], "", Header)
    ;   throw(error(go_wam_template_tags(lowered_function, Path),
                    context(go_render_lowered_function/4, Path)))
    ).

% -- Lowered ITE shell (design §8 Phase 4 slice 1) ---------------------------
%
% Ordered-repeated marker splice (every {{indent}} occurrence listed, like C++
% ite_shell_markers/1) around four already-rendered children: the condition, the
% then branch, the fresh-variable reset and the else branch. No counter: the Go
% ITE uses fixed local names (_trailMark/_condOk/_savedRegs) shadowed per nested
% Go block. Children are spliced as values and never rescanned (a get_constant
% fragment with a '{{Ai}}' atom stays literal). Byte-identical to the old
% inline format/2 block (frozen by the single/sequential/nested-ITE digests).
go_ite_shell_markers([
    "{{indent}}", "{{indent}}", "{{indent}}", "{{indent}}", "{{indent}}",
    "{{indent}}", "{{indent}}",
    "{{condition}}",
    "{{indent}}", "{{indent}}", "{{indent}}",
    "{{then}}",
    "{{indent}}", "{{indent}}", "{{indent}}",
    "{{fresh_reset}}", "{{else}}",
    "{{indent}}", "{{indent}}"
]).

go_ite_shell_values(I, C, T, F, E,
    [I, I, I, I, I, I, I, C, I, I, I, T, I, I, I, F, E, I, I]).

go_render_ite_shell(Indent, Condition, Then, FreshReset, Else, Text) :-
    go_template_root(Root),
    go_render_ite_shell_at_root(Root, Indent, Condition, Then, FreshReset, Else,
                                Text).

go_render_ite_shell_at_root(Root, Indent, Condition, Then, FreshReset, Else,
                            Text) :-
    go_template_path(lowered_ite, Relative),
    directory_file_path(Root, Relative, Path),
    (   go_text_value(Indent), go_text_value(Condition), go_text_value(Then),
        go_text_value(FreshReset), go_text_value(Else)
    ->  true
    ;   throw(error(domain_error(go_wam_ite_variables,
                                 [Indent, Condition, Then, FreshReset, Else]),
                    context(go_render_ite_shell/6, Path)))
    ),
    go_read_asset(Path, lowered_ite, shell, go_render_ite_shell/6, Template),
    go_ite_shell_markers(Markers),
    (   go_split_repeated_markers(Template, Markers, Parts)
    ->  go_text_string(Indent, IndentText),
        maplist(go_text_string, [Condition, Then, FreshReset, Else],
                [CondText, ThenText, FreshText, ElseText]),
        go_ite_shell_values(IndentText, CondText, ThenText, FreshText, ElseText,
                            Values),
        go_interleave_slots(Parts, Values, Joined),
        atomics_to_string(Joined, "", Text)
    ;   throw(error(go_wam_template_tags(lowered_ite, Path),
                    context(go_render_ite_shell/6, Path)))
    ).

% -- Head-match fragment (design §8 Phase 4 slice 2) -------------------------
%
% The shared get_constant/get_integer/get_nil bind-or-match body: one fragment
% with the repeated ordered slots {{I}}×11, {{Comment}}, {{Ai}}, {{GoVal}}×2.
% Spliced exactly like the ITE shell (ordered-repeated markers), NEVER through
% render_template/3: the Comment for a placeholder-shaped atom (get_constant
% '{{Ai}}') contains literal {{Ai}} and sequential key replacement would rescan
% it. Byte-identical to the old three duplicated emit_one/2 bodies (frozen by the
% slice-0 per-fragment digests). GoVal is pre-computed by the emitter (it runs
% the side-effecting intern_atom_go/2 for atoms / nil), so nothing here interns.
go_head_match_markers([
    "{{I}}", "{{Comment}}",
    "{{I}}",
    "{{I}}", "{{Ai}}",
    "{{I}}",
    "{{I}}",
    "{{I}}",
    "{{I}}", "{{GoVal}}",
    "{{I}}", "{{GoVal}}",
    "{{I}}",
    "{{I}}",
    "{{I}}"
]).

go_head_match_values(I, Comment, Ai, GoVal,
    [I, Comment, I, I, Ai, I, I, I, I, GoVal, I, GoVal, I, I, I]).

go_render_head_match(Indent, Comment, Ai, GoVal, Text) :-
    go_template_root(Root),
    go_render_head_match_at_root(Root, Indent, Comment, Ai, GoVal, Text).

go_render_head_match_at_root(Root, Indent, Comment, Ai, GoVal, Text) :-
    go_template_path(head_match, Relative),
    directory_file_path(Root, Relative, Path),
    (   go_head_match_slot(Indent), go_nonempty_text(Comment),
        go_head_match_slot(Ai), go_head_match_nonempty(GoVal)
    ->  true
    ;   throw(error(domain_error(go_wam_head_match_variables,
                                 [Indent, Comment, Ai, GoVal]),
                    context(go_render_head_match/5, Path)))
    ),
    go_read_asset(Path, head_match, fragment, go_render_head_match/5, Template),
    go_head_match_markers(Markers),
    (   go_split_repeated_markers(Template, Markers, Parts)
    ->  go_head_match_values(Indent, Comment, Ai, GoVal, Values),
        go_interleave_slots(Parts, Values, Joined),
        atomics_to_string(Joined, "", Text)
    ;   throw(error(go_wam_template_tags(head_match, Path),
                    context(go_render_head_match/5, Path)))
    ).

% Head-match slots are atomic (the register index Ai is an integer; the indent,
% comment and Go value are atoms/strings). atomics_to_string stringifies each
% exactly as the old format/2 ~w did.
go_head_match_slot(V) :- atom(V), !.
go_head_match_slot(V) :- string(V), !.
go_head_match_slot(V) :- number(V).

go_head_match_nonempty(V) :- go_head_match_slot(V), V \== '', V \== "".

% Split on each marker in order; each marker is matched at its FIRST remaining
% occurrence (ordered-repeated), so {{indent}} can appear many times. Every span
% between markers is linted (no template-origin tag left to emit unresolved).
go_split_repeated_markers(Text, [], [Text]) :-
    go_source_span_clean(Text).
go_split_repeated_markers(Text, [Marker|Markers], [Before|Parts]) :-
    string_length(Marker, Length),
    sub_string(Text, At, Length, _, Marker),
    !,
    sub_string(Text, 0, At, _, Before),
    go_source_span_clean(Before),
    End is At + Length,
    sub_string(Text, End, _, 0, After),
    go_split_repeated_markers(After, Markers, Parts).

go_nonempty_text(Value) :-
    go_text_value(Value),
    Value \== '', Value \== "".

% -- Project files (design §8 Phase 4 slice 3) -------------------------------
%
% Fail-closed variable-render for go.mod/value/instructions/state/main_bench.
% Reads module-relative (no cwd lookup, no silent fallback) then renders through
% the generic engine — byte-identical to the old read_template_file/2 +
% render_template/3, minus the "// Template not found" masking.
go_render_file(Id, Vars, Text) :-
    go_template_root(Root),
    go_render_file_at_root(Root, Id, Vars, Text).

go_render_file_at_root(Root, Id, Vars, Text) :-
    (   go_file_template_path(Id, Relative)
    ->  true
    ;   throw(error(domain_error(go_wam_file_template, Id),
                    context(go_render_file/3, 'unknown Go WAM file template')))
    ),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, Id, file, go_render_file/3, Template),
    render_template(Template, Vars, Text).

% -- Project atom-embedded blocks: spliced, never render_template -------------
%
% Ordered-marker splice (like the ITE/head-match fragments): the package/module
% name and the generated body (imports, predicates, atom table, lowered code) are
% spliced as values and never rescanned. Byte-identical to the old format/2
% headers in write_wam_go_project/3.
go_project_markers(project_lib,
    ["{{package_name}}", "{{imports}}", "{{predicates}}"]).
go_project_markers(project_atoms_with_table,
    ["{{package_name}}", "{{atom_table}}"]).
go_project_markers(project_atoms_runtime_only,
    ["{{package_name}}"]).
go_project_markers(project_lowered,
    ["{{package_name}}", "{{lowered}}"]).
go_project_markers(project_main_parallel,
    ["{{module_name}}",
     "{{package_name}}", "{{package_name}}", "{{package_name}}",
     "{{package_name}}", "{{package_name}}", "{{package_name}}"]).

go_render_project_block(Id, Values, Text) :-
    go_template_root(Root),
    go_render_project_block_at_root(Root, Id, Values, Text).

go_render_project_block_at_root(Root, Id, Values, Text) :-
    (   go_project_path(Id, Relative)
    ->  true
    ;   throw(error(domain_error(go_wam_project_block, Id),
                    context(go_render_project_block/3, 'unknown Go WAM project block')))
    ),
    (   maplist(go_head_match_slot, Values)
    ->  true
    ;   throw(error(domain_error(go_wam_project_block_variables, Values),
                    context(go_render_project_block/3, Id)))
    ),
    directory_file_path(Root, Relative, Path),
    go_read_asset(Path, Id, project, go_render_project_block/3, Template),
    go_project_markers(Id, Markers),
    (   go_split_repeated_markers(Template, Markers, Parts)
    ->  go_interleave_slots(Parts, Values, Joined),
        atomics_to_string(Joined, "", Text)
    ;   throw(error(go_wam_template_tags(Id, Path),
                    context(go_render_project_block/3, Id)))
    ).

% Named wrappers (the emitter calls these) -----------------------------------
go_render_project_lib_header(PackageName, Imports, Predicates, Text) :-
    go_render_project_block(project_lib, [PackageName, Imports, Predicates], Text).
go_render_project_lib_header_at_root(Root, PackageName, Imports, Predicates, Text) :-
    go_render_project_block_at_root(Root, project_lib,
        [PackageName, Imports, Predicates], Text).

go_render_project_atoms_with_table(PackageName, AtomTable, Text) :-
    go_render_project_block(project_atoms_with_table, [PackageName, AtomTable], Text).
go_render_project_atoms_with_table_at_root(Root, PackageName, AtomTable, Text) :-
    go_render_project_block_at_root(Root, project_atoms_with_table,
        [PackageName, AtomTable], Text).

go_render_project_atoms_runtime_only(PackageName, Text) :-
    go_render_project_block(project_atoms_runtime_only, [PackageName], Text).
go_render_project_atoms_runtime_only_at_root(Root, PackageName, Text) :-
    go_render_project_block_at_root(Root, project_atoms_runtime_only,
        [PackageName], Text).

go_render_project_lowered_header(PackageName, Lowered, Text) :-
    go_render_project_block(project_lowered, [PackageName, Lowered], Text).
go_render_project_lowered_header_at_root(Root, PackageName, Lowered, Text) :-
    go_render_project_block_at_root(Root, project_lowered,
        [PackageName, Lowered], Text).

go_render_project_main_parallel(ModuleName, PackageName, Text) :-
    go_render_project_block(project_main_parallel,
        [ModuleName, PackageName, PackageName, PackageName,
         PackageName, PackageName, PackageName], Text).
go_render_project_main_parallel_at_root(Root, ModuleName, PackageName, Text) :-
    go_render_project_block_at_root(Root, project_main_parallel,
        [ModuleName, PackageName, PackageName, PackageName,
         PackageName, PackageName, PackageName], Text).

% -- Preflight (design §6.2: runs BEFORE make_directory_path/1) --------------
%
% Loads + lints every required asset and renders the shell once with
% representative values. A broken template fails here, leaving no directory.
go_preflight :-
    go_render_helper_methods(_),
    go_render_step_method(_),
    go_render_template(runtime_shell,
        [package_name="wam", step_method="// preflight"], _),
    go_render_lowered_function("goPreflightFn",
        "// goPreflightFn — lowered from preflight/1", "", _),
    go_render_ite_shell("", "", "", "", "", _),
    go_render_head_match("    ", "get_constant foo, A1", 0, "goPreflightAtom", _),
    % Project files + atom-embedded blocks (slice 3).
    go_render_file(go_mod, [module_name="uw-preflight"], _),
    go_render_file(value, [package_name="wam", date="preflight"], _),
    go_render_file(instructions, [package_name="wam", date="preflight"], _),
    go_render_file(state, [package_name="wam", date="preflight"], _),
    go_render_file(main_bench, [package_name=main], _),
    go_render_project_lib_header("wam", "", "", _),
    go_render_project_atoms_with_table("wam", "", _),
    go_render_project_atoms_runtime_only("wam", _),
    go_render_project_lowered_header("wam", "", _),
    go_render_project_main_parallel("uw-preflight", "wam", _).

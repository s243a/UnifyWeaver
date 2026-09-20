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
    go_preflight/0,                % load + validate + render every required asset once
    go_with_template_root_for_test/2
]).

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(filesex), [directory_file_path/3]).

:- thread_local test_go_template_root/1.

% -- Registry: template ID -> path relative to templates/targets/go_wam ------

% Shells carry ordered variable/child slots; sections are variable-free static
% Go spliced verbatim after a Go-aware tag lint.
go_template_path(runtime_shell,   'runtime.go.mustache').

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

% -- Preflight (design §6.2: runs BEFORE make_directory_path/1) --------------
%
% Loads + lints every required asset and renders the shell once with
% representative values. A broken template fails here, leaving no directory.
go_preflight :-
    go_render_helper_methods(_),
    go_render_template(runtime_shell,
        [package_name="wam", step_method="// preflight"], _).

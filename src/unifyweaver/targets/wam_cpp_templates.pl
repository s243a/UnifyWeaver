:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_cpp_templates, [cpp_render_template/3, cpp_render_template_at_root/4]).

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module('../core/template_system', [render_template/3]).

cpp_template_path(runtime_header, 'runtime.h.mustache').

cpp_render_template(Id, Vars, Text) :-
    source_file(cpp_render_template(_, _, _), Source),
    file_directory_name(Source, ModuleDir),
    directory_file_path(ModuleDir, '../../../templates/targets/cpp_wam', Root),
    cpp_render_template_at_root(Root, Id, Vars, Text).

% Explicit root is for isolated template fixtures; emitters use the module root.
cpp_render_template_at_root(Root, Id, Vars, Text) :-
    (   cpp_template_path(Id, Name)
    ->  true
    ;   throw(error(domain_error(cpp_wam_template, Id),
                    context(cpp_render_template/3, 'unknown C++ WAM template')))
    ),
    directory_file_path(Root, Name, Path),
    (   Vars == []
    ->  true
    ;   throw(error(domain_error(cpp_wam_template_variables, Vars),
                    context(cpp_render_template/3, Id)))
    ),
    catch(read_file_to_string(Path, Template, [encoding(utf8)]),
          Error,
          throw(error(cpp_wam_template_load(Id, Path, Error),
                      context(cpp_render_template/3, runtime_header)))),
    (   Template == ""
    ->  throw(error(cpp_wam_template_empty(Id, Path),
                    context(cpp_render_template/3, runtime_header)))
    ;   true
    ),
    % The header has no substitution keys. Render through the existing engine,
    % while keeping literal C++ braces and source text untouched.
    render_template(Template, [], Text).

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pattern_stache_whitespace.pl - line and byte fidelity of the
% production pattern_stache engine (dialect version 1).
%
% Run: swipl -g run_tests -t halt tests/core/test_pattern_stache_whitespace.pl
%
% WHY THIS SUITE EXISTS.  Both witnessed v1 consumers normalized
% whitespace per rendering (normalize_space/2), so nothing pinned what
% the engine does to bytes it is handed — including the three
% properties the SPEC's Whitespace section already declares normative
% (bodies are literal, nothing is trimmed, the caller owns policy).
% This suite is their home.
%
% THIS IS NOT WHITESPACE-EXACTNESS MACHINERY, and no prospective
% consumer asked for any.  plawk (see
% prototypes/mu_cosine/RECORD_prospective_consumer_plawk.md) needs
% marker-adjacent whitespace to be CONTROLLABLE — byte-identity is a
% property of their regression tool, not of this dialect, and they
% explicitly asked that nothing be built on its account.  The
% marker_adjacency unit below answers the controllability question;
% the rest characterizes behaviour so a change to it shows up in a
% diff rather than downstream.
%
% Three rows are HAZARDS, not features, and are marked so at the test:
% the preamble discard was unmentioned in the SPEC until now, unknown
% keys are silently left verbatim by design, and interpolated values
% are rescanned by later dict keys.

:- use_module('../../src/unifyweaver/core/pattern_stache').
:- use_module(library(plunit)).

make_stache_file(Text, Path) :-
    tmp_file(pattern_stache_ws_test, Base),
    atom_concat(Base, '.stache', Path),
    setup_call_cleanup(
        open(Path, write, S),
        write(S, Text),
        close(S)).

header("{{! dialect(pattern_stache, 1) }}\n").

%% ============================================
%% Bodies are literal (SPEC, Whitespace — normative)
%% ============================================

:- begin_tests(literal_bodies).

test(indentation_and_newlines_preserved) :-
    render_stache(stache(1, "{{match k}}{{case a}}L1\n  L2\n    L3\n{{/match}}"),
                  [k=a], R),
    R == "L1\n  L2\n    L3\n".

test(no_trim_of_leading_or_trailing_space) :-
    render_stache(stache(1, "{{match k}}{{case a}}   padded   {{/match}}"),
                  [k=a], R),
    R == "   padded   ".

test(blank_lines_inside_a_body_survive) :-
    render_stache(stache(1, "{{match k}}{{case a}}L1\n\n\nL2{{/match}}"),
                  [k=a], R),
    R == "L1\n\n\nL2".

test(crlf_bytes_are_not_normalized) :-
    render_stache(stache(1, "{{match k}}{{case a}}L1\r\nL2\r\n{{/match}}"),
                  [k=a], R),
    R == "L1\r\nL2\r\n".

test(tabs_survive) :-
    render_stache(stache(1, "{{match k}}{{case a}}\tx\ty{{/match}}"), [k=a], R),
    R == "\tx\ty".

test(text_outside_the_block_is_verbatim_on_both_sides) :-
    render_stache(stache(1, "  PRE\n{{match k}}{{case a}}B{{/match}}\nPOST  "),
                  [k=a], R),
    R == "  PRE\nB\nPOST  ".

test(no_match_no_default_renders_empty_without_disturbing_neighbours) :-
    render_stache(stache(1, "PRE\n{{match k}}{{case a}}B{{/match}}\nPOST"),
                  [k=zzz], R),
    R == "PRE\n\nPOST".

:- end_tests(literal_bodies).

%% ============================================
%% Load-time byte effects
%% ============================================

:- begin_tests(load_fidelity).

test(header_line_removed_rest_of_file_verbatim) :-
    header(H),
    string_concat(H, "  A\n\tB\n\nC\n", Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, stache(1, Body)),
    Body == "  A\n\tB\n\nC\n".

test(blank_lines_before_the_header_are_dropped) :-
    header(H),
    string_concat("\n\n   \n", H, Prefix),
    string_concat(Prefix, "X\n", Text),
    make_stache_file(Text, Path),
    load_stache_file(Path, stache(1, Body)),
    Body == "X\n".

:- end_tests(load_fidelity).

%% ============================================
%% HAZARD: the preamble discard
%% ============================================
%
% Text between {{match k}} and the FIRST {{case}} is discarded — not
% rendered, not an error.  The SPEC was silent on this until the
% prospective-consumer record; it is what lets a template put the match
% tag on its own line without emitting that newline, and it is also the
% one place the engine deletes bytes it was handed.

:- begin_tests(preamble_discard).

test(preamble_text_is_discarded) :-
    render_stache(stache(1, "PRE{{match k}}JUNK{{case a}}BODY{{/match}}POST"),
                  [k=a], R),
    R == "PREBODYPOST".

test(newline_after_the_match_tag_is_discarded) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}BODY{{/match}}"), [k=a], R),
    R == "BODY".

test(discard_applies_on_the_default_path_too) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}A{{default}}D{{/match}}"),
                  [k=zzz], R),
    R == "D".

:- end_tests(preamble_discard).

%% ============================================
%% Marker adjacency: is the whitespace controllable?
%% ============================================
%
% The question a prospective consumer actually has (plawk record,
% requirement 3): can a template author reach an exact target text, or
% does the dialect impose whitespace it cannot control?  Answer: every
% marker-adjacent newline is the author's, but only {{match}} gets a
% free line.  The other three markers are literal boundaries, so a tag
% must share a line with the body character next to it.  That is a
% layout cost, not a capability limit — and it is exactly what
% standalone-line semantics would remove (SPEC, exclusions).

:- begin_tests(marker_adjacency).

test(case_tag_sharing_its_line_reaches_the_target_exactly) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}X\n{{/match}}"), [k=a], R),
    R == "X\n".

test(case_tag_on_its_own_line_costs_a_leading_newline) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}\nX\n{{/match}}"), [k=a], R),
    R == "\nX\n".

test(close_tag_on_its_own_line_costs_a_trailing_newline) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}X\n{{/match}}\n"), [k=a], R),
    R == "X\n\n".

test(default_tag_obeys_the_same_rule) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}A{{default}}D\n{{/match}}"), [k=zzz], R),
    R == "D\n".

test(default_tag_on_its_own_line_costs_a_leading_newline) :-
    render_stache(stache(1, "{{match k}}\n{{case a}}A{{default}}\nD\n{{/match}}"), [k=zzz], R),
    R == "\nD\n".

test(consecutive_cases_compose_without_extra_cost) :-
    T = "{{match k}}\n{{case a}}A1\nA2\n{{case b}}B1\n{{/match}}",
    render_stache(stache(1, T), [k=a], Ra),
    render_stache(stache(1, T), [k=b], Rb),
    Ra == "A1\nA2\n",
    Rb == "B1\n".

test(a_block_may_occupy_whole_lines_without_contributing_any) :-
    render_stache(stache(1, "PRE\n{{match k}}\n{{case a}}X\n{{/match}}POST"), [k=a], R),
    R == "PRE\nX\nPOST".

:- end_tests(marker_adjacency).

%% ============================================
%% HAZARD: silent verbatim on an unbound key
%% ============================================
%
% SPEC, deliberate exclusions: "missing-key / unbound-placeholder
% changes — left verbatim, matching template_system.pl", revisit when
% "a consumer needs fail-on-unbound rendering".  The failure has no
% local symptom — the render succeeds and the output is well-formed
% text that happens to contain a template marker.  Pinned as a property
% of the engine; NO consumer has claimed this row, prospective or
% otherwise, and the revisit condition stays unclaimed.

:- begin_tests(unbound_is_silent).

test(unknown_key_is_left_verbatim_and_does_not_throw) :-
    render_stache(stache(1, "%end_str_ptr_{{Idx}}"), ['Idxx'=3], R),
    R == "%end_str_ptr_{{Idx}}".

test(key_spelling_is_exact_lowercase_key_does_not_fill_uppercase_slot) :-
    render_stache(stache(1, "%p_{{Idx}}"), [idx=3], R),
    R == "%p_{{Idx}}".

test(quoted_uppercase_key_does_fill_it) :-
    render_stache(stache(1, "%p_{{Idx}}"), ['Idx'=3], R),
    R == "%p_3".

test(q_form_is_also_left_verbatim_when_unbound) :-
    render_stache(stache(1, "{{q:Idx}}"), [], R),
    R == "{{q:Idx}}".

:- end_tests(unbound_is_silent).

%% ============================================
%% HAZARD: substitution is a sequential global replace
%% ============================================
%
% substitute_placeholders/3 walks the dict in order, replacing every
% occurrence of each key's marker in the whole text.  A value that
% itself contains a later key's marker is therefore rescanned; the
% result depends on dict order.  Not a defect of v1 — a property a
% consumer that interpolates untrusted text has to know.

:- begin_tests(substitution_order).

test(value_containing_a_later_keys_marker_is_rescanned) :-
    render_stache(stache(1, "A={{a}} B={{b}}"), [a='{{b}}', b=zzz], R),
    R == "A=zzz B=zzz".

test(the_same_dict_in_the_other_order_does_not_rescan) :-
    render_stache(stache(1, "A={{a}} B={{b}}"), [b=zzz, a='{{b}}'], R),
    R == "A={{b}} B=zzz".

:- end_tests(substitution_order).

%% ============================================
%% HAZARD: a case binding shadows a threaded dict key
%% ============================================
%
% SPEC, matching semantics 4: bindings are prepended and shadow outer
% keys "silently and lexically".  A caller threading an index through
% every render loses it inside any case whose pattern names a variable
% with the same spelling — decided per case, affecting a property that
% spans the whole emission.

:- begin_tests(binding_shadowing).

test(pattern_variable_shadows_the_threaded_key) :-
    render_stache(stache(1, "{{match slot}}{{case boxed(Idx)}}%inner_{{Idx}}{{/match}}"),
                  [slot=boxed(99), 'Idx'=3], R),
    R == "%inner_99".

test(the_outer_key_is_untouched_after_the_block) :-
    render_stache(stache(1, "{{match slot}}{{case boxed(Idx)}}%inner_{{Idx}}{{/match}} outer={{Idx}}"),
                  [slot=boxed(99), 'Idx'=3], R),
    R == "%inner_99 outer=3".

:- end_tests(binding_shadowing).

%% ============================================
%% Value spelling
%% ============================================
%
% {{Key}} is ~w and {{q:Key}} is ~q, so interpolated numbers inherit
% SWI's writer.  Where that diverges from another producer's spelling
% is measured in prototypes/mu_cosine/pattern_stache/pe_number.pl; a
% consumer diffing against golden bytes produced elsewhere needs that
% module, not this one.

:- begin_tests(value_spelling).

test(integer_index) :-
    render_stache(stache(1, "i={{i}}"), [i=3], R),
    R == "i=3".

test(float_keeps_its_point_zero) :-
    render_stache(stache(1, "i={{i}}"), [i=3.0], R),
    R == "i=3.0".

test(w_does_not_quote_and_q_does) :-
    render_stache(stache(1, "w={{v}} q={{q:v}}"), [v='has space'], R),
    R == "w=has space q='has space'".

:- end_tests(value_spelling).

%% ============================================
%% The prospective consumer's own shape, end to end
%% ============================================
%
% Requirement 2 of the plawk record: a conditional list of lines whose
% membership depends on a resolved slot kind, with a caller-threaded
% index woven into every generated name.  Driver iterates, template
% dispatches, index arrives in the dict.

:- begin_tests(threaded_index_shape).

slot_template(
"{{match slot}}\n\c
{{case str_slot(Name)}}  %{{Name}}_ptr_{{Idx}} = getelementptr i8, i8* %buf, i64 0\n\c
  %{{Name}}_len_{{Idx}} = call i64 @strlen(i8* %{{Name}}_ptr_{{Idx}})\n\c
  %end_str_ptr_{{Idx}} = getelementptr i8, i8* %{{Name}}_ptr_{{Idx}}, i64 %{{Name}}_len_{{Idx}}\n\c
{{case num_slot(Name)}}  %{{Name}}_val_{{Idx}} = load double, double* %slotbuf\n\c
  %end_num_{{Idx}} = fptosi double %{{Name}}_val_{{Idx}} to i64\n\c
{{/match}}").

test(string_slot_emits_its_three_lines_with_the_threaded_index) :-
    slot_template(T),
    render_stache(stache(1, T), [slot=str_slot(head), 'Idx'=3], R),
    R == "  %head_ptr_3 = getelementptr i8, i8* %buf, i64 0\n\c
  %head_len_3 = call i64 @strlen(i8* %head_ptr_3)\n\c
  %end_str_ptr_3 = getelementptr i8, i8* %head_ptr_3, i64 %head_len_3\n".

test(numeric_slot_emits_a_different_list) :-
    slot_template(T),
    render_stache(stache(1, T), [slot=num_slot(tail), 'Idx'=7], R),
    R == "  %tail_val_7 = load double, double* %slotbuf\n\c
  %end_num_7 = fptosi double %tail_val_7 to i64\n".

test(the_driver_concatenates_across_slots_the_template_does_not_iterate) :-
    slot_template(T),
    findall(R,
            ( member(Slot-I, [str_slot(a)-0, num_slot(b)-1, str_slot(c)-2]),
              render_stache(stache(1, T), [slot=Slot, 'Idx'=I], R)
            ),
            Rs),
    atomics_to_string(Rs, Out),
    % every generated name carries its own index; no name collides
    aggregate_all(count, sub_string(Out, _, _, _, "%end_str_ptr_"), 2),
    contains(Out, "%end_str_ptr_0"),
    contains(Out, "%end_num_1"),
    contains(Out, "%end_str_ptr_2").

contains(String, Sub) :- once(sub_string(String, _, _, _, Sub)).

:- end_tests(threaded_index_shape).

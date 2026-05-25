:- encoding(utf8).
% Isolated edge case tests for match/case template feature.
% Run: swipl -q -g run_tests -t halt tests/test_template_match_case.pl

:- use_module('../src/unifyweaver/core/template_system').

:- dynamic test_count/2.  % test_count(pass, N), test_count(fail, N)

run_tests :-
    retractall(test_count(_, _)),
    assertz(test_count(pass, 0)),
    assertz(test_count(fail, 0)),
    writeln('=== Match/Case Edge Case Tests ==='),

    run_test("Nested match: outer=a inner=y",
        render_template("{{match outer}}{{case a}}[{{match inner}}{{case x}}AX{{case y}}AY{{/match}}]{{case b}}B{{/match}}",
                         [outer=a, inner=y], R1),
        sub_string(R1, _, _, _, "[AY]")),

    run_test("Nested match: outer=a inner=x",
        render_template("{{match outer}}{{case a}}[{{match inner}}{{case x}}AX{{case y}}AY{{/match}}]{{case b}}B{{/match}}",
                         [outer=a, inner=x], R2),
        sub_string(R2, _, _, _, "[AX]")),

    run_test("Nested match: outer=b (skip nested)",
        render_template("{{match outer}}{{case a}}{{match inner}}{{case x}}AX{{/match}}{{case b}}B{{/match}}",
                         [outer=b], R3),
        sub_string(R3, _, _, _, "B")),

    run_test("Nested match: outer=a inner unmatched -> default",
        render_template("{{match outer}}{{case a}}{{match inner}}{{case x}}X{{default}}DEF{{/match}}{{case b}}B{{/match}}",
                         [outer=a, inner=z], R4),
        sub_string(R4, _, _, _, "DEF")),

    run_test("Double nested match",
        render_template("{{match l1}}{{case a}}{{match l2}}{{case p}}{{match l3}}{{case q}}DEEP{{/match}}{{/match}}{{/match}}",
                         [l1=a, l2=p, l3=q], R5),
        sub_string(R5, _, _, _, "DEEP")),

    run_test("Section inside case body (truthy)",
        render_template("{{match mode}}{{case cached}}{{#l2}}L2{{/l2}}{{^l2}}noL2{{/l2}}{{/match}}",
                         [mode=cached, l2=true], R6),
        (sub_string(R6, _, _, _, "L2"), \+ sub_string(R6, _, _, _, "noL2"))),

    run_test("Section inside case body (falsy)",
        render_template("{{match mode}}{{case cached}}{{#l2}}L2{{/l2}}{{^l2}}noL2{{/l2}}{{/match}}",
                         [mode=cached, l2=false], R7),
        (sub_string(R7, _, _, _, "noL2"))),

    run_test("Hyphens in case value",
        render_template("{{match t}}{{case wam-fsharp}}FS{{case wam-haskell}}HS{{/match}}",
                         [t='wam-fsharp'], R8),
        sub_string(R8, _, _, _, "FS")),

    run_test("Dots in case value",
        render_template("{{match ver}}{{case 3.11}}old{{case 3.12}}new{{/match}}",
                         [ver='3.12'], R9),
        sub_string(R9, _, _, _, "new")),

    run_test("Underscores in case value",
        render_template("{{match b}}{{case lmdb_offset}}LMDB{{case sorted_array}}SA{{/match}}",
                         [b=sorted_array], R10),
        sub_string(R10, _, _, _, "SA")),

    run_test("Empty case body (skip)",
        render_template("{{match m}}{{case skip}}{{case keep}}K{{/match}}", [m=skip], R11),
        R11 = ""),

    run_test("Empty case body (keep)",
        render_template("{{match m}}{{case skip}}{{case keep}}K{{/match}}", [m=keep], R12),
        sub_string(R12, _, _, _, "K")),

    run_test("Only default",
        render_template("{{match any}}{{default}}ALWAYS{{/match}}", [any=whatever], R13),
        sub_string(R13, _, _, _, "ALWAYS")),

    run_test("Only default, key missing from dict",
        render_template("{{match missing}}{{default}}FALLBACK{{/match}}", [], R14),
        sub_string(R14, _, _, _, "FALLBACK")),

    run_test("No match no default -> empty",
        render_template("{{match m}}{{case a}}A{{case b}}B{{/match}}", [m=c], R15),
        R15 = ""),

    run_test("Whitespace preserved around case body",
        render_template("{{match m}}\n{{case a}}\n  BODY\n{{/match}}", [m=a], R16),
        sub_string(R16, _, _, _, "\n  BODY\n")),

    run_test("Variable substitution inside case",
        render_template("{{match m}}{{case x}}val={{v}}{{/match}}", [m=x, v='42'], R17),
        sub_string(R17, _, _, _, "val=42")),

    run_test("Multiple match blocks in one template",
        render_template("{{match a}}{{case x}}AX{{/match}}-{{match b}}{{case y}}BY{{/match}}",
                         [a=x, b=y], R18),
        sub_string(R18, _, _, _, "AX-BY")),

    run_test("Malformed: no closing /match",
        render_template("{{match k}}{{case a}}body", [k=a], R19),
        sub_string(R19, _, _, _, "{{match k}}")),

    run_test("Malformed: no key in match tag",
        render_template("{{match}}{{case a}}body{{/match}}", [], R20),
        sub_string(R20, _, _, _, "{{match}}")),

    run_test("Malformed: orphan /match",
        render_template("X{{/match}}Y", [], R21),
        (sub_string(R21, _, _, _, "X"), sub_string(R21, _, _, _, "Y"))),

    run_test("Malformed: empty case value",
        render_template("{{match k}}{{case }}body{{/match}}", [k=a], _R22),
        true),  % should not crash

    % Summary
    test_count(pass, P),
    test_count(fail, F),
    Total is P + F,
    format('~n=== Results: ~w/~w passed ===~n', [P, Total]),
    (F > 0 -> halt(1) ; true).

run_test(Name, Setup, Check) :-
    write(Name), write(': '),
    (   catch(
            (call(Setup), (call(Check) -> true ; fail)),
            Err,
            (format('ERROR: ~w~n', [Err]), record_fail, fail)
        )
    ->  writeln('PASS'), record_pass
    ;   writeln('FAIL'), record_fail
    ).

record_pass :-
    test_count(pass, N),
    retract(test_count(pass, N)),
    N1 is N + 1,
    assertz(test_count(pass, N1)).

record_fail :-
    test_count(fail, N),
    retract(test_count(fail, N)),
    N1 is N + 1,
    assertz(test_count(fail, N1)).

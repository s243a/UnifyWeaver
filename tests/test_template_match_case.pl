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

    % === Section 3.9: Real template file loaded from disk ===

    run_test("3.9a File load: ST mode + lmdb backend",
        (   load_template_from_file("tests/fixtures/kernel_match_dispatch.hs.mustache", Tmpl23),
            render_template(Tmpl23, [module_name='TestModule', run_mode=st,
                                     backend=lmdb_offset, run_function=runMutableRegs,
                                     has_metrics=true, query_expr='pure 42'], R23)
        ),
        (   sub_string(R23, _, _, _, "module=TestModule"),
            sub_string(R23, _, _, _, "import Control.Monad.ST"),
            sub_string(R23, _, _, _, "import LmdbFactSource"),
            sub_string(R23, _, _, _, "st_result="),
            sub_string(R23, _, _, _, "metrics=enabled"),
            \+ sub_string(R23, _, _, _, "pure mode"),
            \+ sub_string(R23, _, _, _, "import SortedArraySource")
        )),

    run_test("3.9b File load: pure mode + memory backend",
        (   load_template_from_file("tests/fixtures/kernel_match_dispatch.hs.mustache", Tmpl24),
            render_template(Tmpl24, [module_name='PureMod', run_mode=pure,
                                     backend=memory, run_function=run,
                                     has_metrics=false, query_expr='length [1,2,3]'], R24)
        ),
        (   sub_string(R24, _, _, _, "module=PureMod"),
            sub_string(R24, _, _, _, "pure mode"),
            sub_string(R24, _, _, _, "in-memory backend"),
            sub_string(R24, _, _, _, "pure_result="),
            sub_string(R24, _, _, _, "length [1,2,3]"),
            sub_string(R24, _, _, _, "metrics=disabled"),
            \+ sub_string(R24, _, _, _, "import Control.Monad.ST"),
            \+ sub_string(R24, _, _, _, "import LmdbFactSource")
        )),

    run_test("3.9c File load: unknown run_mode -> default",
        (   load_template_from_file("tests/fixtures/kernel_match_dispatch.hs.mustache", Tmpl25),
            render_template(Tmpl25, [module_name='FallbackMod', run_mode=exotic,
                                     backend=sorted_array, run_function=run,
                                     has_metrics=true, query_expr='0'], R25)
        ),
        (   sub_string(R25, _, _, _, "unknown run_mode"),
            sub_string(R25, _, _, _, "fallback_mode"),
            sub_string(R25, _, _, _, "import SortedArraySource"),
            \+ sub_string(R25, _, _, _, "import Control.Monad.ST")
        )),

    run_test("3.9d File load: backend not in dict -> default empty",
        (   load_template_from_file("tests/fixtures/kernel_match_dispatch.hs.mustache", Tmpl26),
            render_template(Tmpl26, [module_name='NoBackend', run_mode=pure,
                                     run_function=run, has_metrics=false,
                                     query_expr='0'], R26)
        ),
        (   sub_string(R26, _, _, _, "module=NoBackend"),
            \+ sub_string(R26, _, _, _, "import LmdbFactSource"),
            \+ sub_string(R26, _, _, _, "import SortedArraySource")
        )),

    % === Section 3.10: Integration with real codegen template ===
    % Uses the actual program.fs.mustache from the fsharp_wam target.

    run_test("3.10a Codegen: LMDB cached materialisation",
        (   load_template_from_file("templates/targets/fsharp_wam/program.fs.mustache", TmplFsCached),
            render_template(TmplFsCached,
                [foreign_preds = '"category_ancestor/4"',
                 lookup_sources_expr = 'Map.empty',
                 has_csr = false, has_lmdb = true,
                 materialisation = cached, l2_capacity = '"auto"'], RFsCached)
        ),
        (   sub_string(RFsCached, _, _, _, "open LmdbFactSource"),
            sub_string(RFsCached, _, _, _, "TwoLevelCachedLookupSource"),
            sub_string(RFsCached, _, _, _, "LmdbCursorLookup"),
            \+ sub_string(RFsCached, _, _, _, "DictLookupSource"),
            \+ sub_string(RFsCached, _, _, _, "open CsrReader")
        )),

    run_test("3.10b Codegen: LMDB eager materialisation",
        (   load_template_from_file("templates/targets/fsharp_wam/program.fs.mustache", TmplFsEager),
            render_template(TmplFsEager,
                [foreign_preds = '"category_ancestor/4"',
                 lookup_sources_expr = 'Map.empty',
                 has_csr = false, has_lmdb = true,
                 materialisation = eager, l2_capacity = '"auto"'], RFsEager)
        ),
        (   sub_string(RFsEager, _, _, _, "DictLookupSource"),
            sub_string(RFsEager, _, _, _, "loadDupsortRelationDict"),
            \+ sub_string(RFsEager, _, _, _, "TwoLevelCachedLookupSource")
        )),

    run_test("3.10c Codegen: LMDB lazy materialisation",
        (   load_template_from_file("templates/targets/fsharp_wam/program.fs.mustache", TmplFsLazy),
            render_template(TmplFsLazy,
                [foreign_preds = '"category_ancestor/4"',
                 lookup_sources_expr = 'Map.empty',
                 has_csr = false, has_lmdb = true,
                 materialisation = lazy, l2_capacity = '"auto"'], RFsLazy)
        ),
        (   sub_string(RFsLazy, _, _, _, "LmdbCursorLookup"),
            sub_string(RFsLazy, _, _, _, "Lazy"),
            \+ sub_string(RFsLazy, _, _, _, "TwoLevelCachedLookupSource"),
            \+ sub_string(RFsLazy, _, _, _, "DictLookupSource")
        )),

    run_test("3.10d Codegen: no LMDB (has_lmdb=false), no match block output",
        (   load_template_from_file("templates/targets/fsharp_wam/program.fs.mustache", TmplFsNoLmdb),
            render_template(TmplFsNoLmdb,
                [foreign_preds = '', lookup_sources_expr = 'Map.empty',
                 has_csr = false, has_lmdb = false,
                 materialisation = cached, l2_capacity = '"auto"'], RFsNoLmdb)
        ),
        (   \+ sub_string(RFsNoLmdb, _, _, _, "open LmdbFactSource"),
            \+ sub_string(RFsNoLmdb, _, _, _, "lmdbFactSource"),
            \+ sub_string(RFsNoLmdb, _, _, _, "TwoLevelCachedLookupSource"),
            sub_string(RFsNoLmdb, _, _, _, "WcLookupSources")
        )),

    run_test("3.10e Codegen: CSR + LMDB together",
        (   load_template_from_file("templates/targets/fsharp_wam/program.fs.mustache", TmplFsBoth),
            render_template(TmplFsBoth,
                [foreign_preds = '"category_ancestor/4"',
                 lookup_sources_expr = 'Map.ofList [ ("category_child", csrSrc) ]',
                 has_csr = true, has_lmdb = true,
                 materialisation = eager, l2_capacity = '"auto"'], RFsBoth)
        ),
        (   sub_string(RFsBoth, _, _, _, "open CsrReader"),
            sub_string(RFsBoth, _, _, _, "open LmdbFactSource"),
            sub_string(RFsBoth, _, _, _, "DictLookupSource")
        )),

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

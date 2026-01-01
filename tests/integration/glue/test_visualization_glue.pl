% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Integration Tests for Visualization Glue Modules
%
% Tests: graph_generator, curve_plot_generator, matplotlib_generator

:- use_module('../../../src/unifyweaver/glue/graph_generator').
:- use_module('../../../src/unifyweaver/glue/curve_plot_generator').
:- use_module('../../../src/unifyweaver/glue/matplotlib_generator').

:- dynamic test_passed/0.
:- dynamic test_failed/0.
:- dynamic test_count/1.

% ============================================================================
% TEST RUNNER
% ============================================================================

run_tests :-
    retractall(test_passed),
    retractall(test_failed),
    retractall(test_count(_)),
    assertz(test_count(0)),

    format('========================================~n'),
    format('Visualization Glue Integration Tests~n'),
    format('========================================~n~n'),

    % Graph Generator Tests
    run_graph_generator_tests,

    % Curve Plot Generator Tests
    run_curve_plot_generator_tests,

    % Matplotlib Generator Tests
    run_matplotlib_generator_tests,

    % Summary
    print_summary.

print_summary :-
    aggregate_all(count, test_passed, Passed),
    aggregate_all(count, test_failed, Failed),
    Total is Passed + Failed,
    format('~n========================================~n'),
    format('Results: ~w/~w tests passed~n', [Passed, Total]),
    (Failed =:= 0
    ->  format('All tests passed!~n')
    ;   format('~w tests failed!~n', [Failed])
    ),
    format('========================================~n').

% ============================================================================
% GRAPH GENERATOR TESTS
% ============================================================================

run_graph_generator_tests :-
    format('~n--- Graph Generator Tests ---~n'),

    % Node queries
    test("Has family tree nodes", (
        all_nodes(Nodes1),
        length(Nodes1, NodeCount),
        NodeCount > 8
    )),

    test("Has specific nodes", (
        graph_node(abraham, _),
        graph_node(isaac, _),
        graph_node(jacob, _)
    )),

    % Edge queries
    test("Has family tree edges", (
        all_edges(Edges1),
        length(Edges1, EdgeCount),
        EdgeCount > 6
    )),

    test("Abraham is parent of Isaac", (
        graph_edge(abraham, isaac, EdgeProps),
        member(relation(parent), EdgeProps)
    )),

    % Graph spec
    test("family_tree spec exists", (
        graph_spec(family_tree, FamilyConfig),
        member(title(_), FamilyConfig)
    )),

    test("simple_graph spec exists", (
        graph_spec(simple_graph, _)
    )),

    % Component generation
    test("Generate React component", (
        generate_graph_component(family_tree, CompCode1),
        atom_length(CompCode1, CompLen),
        CompLen > 3000
    )),

    test("Component has React imports", (
        generate_graph_component(family_tree, CompCode2),
        sub_atom(CompCode2, _, _, _, 'import React')
    )),

    test("Component has Cytoscape import", (
        generate_graph_component(family_tree, CompCode3),
        sub_atom(CompCode3, _, _, _, 'cytoscape')
    )),

    test("Component has TypeScript interface", (
        generate_graph_component(family_tree, CompCode4),
        sub_atom(CompCode4, _, _, _, 'interface')
    )),

    % Data generation
    test("Graph data contains nodes", (
        generate_graph_data(family_tree, GraphData1),
        sub_atom(GraphData1, _, _, _, 'abraham')
    )),

    test("Graph data contains edges", (
        generate_graph_data(family_tree, GraphData2),
        sub_atom(GraphData2, _, _, _, 'source')
    )),

    % CSS generation
    test("CSS has container class", (
        generate_graph_styles(family_tree, GraphCss1),
        sub_atom(GraphCss1, _, _, _, '.container')
    )),

    test("CSS has graph class", (
        generate_graph_styles(family_tree, GraphCss2),
        sub_atom(GraphCss2, _, _, _, '.graph')
    )),

    % Cytoscape config
    test("Config has node selector", (
        generate_cytoscape_config(family_tree, CytoConfig1),
        sub_atom(CytoConfig1, _, _, _, 'selector')
    )),

    test("Config has layout", (
        generate_cytoscape_config(family_tree, CytoConfig2),
        sub_atom(CytoConfig2, _, _, _, 'layout')
    )).

% ============================================================================
% CURVE PLOT GENERATOR TESTS
% ============================================================================

run_curve_plot_generator_tests :-
    format('~n--- Curve Plot Generator Tests ---~n'),

    % Curve queries
    test("Has multiple curves defined", (
        all_curves(CurveList),
        length(CurveList, CurveCount),
        CurveCount > 10
    )),

    test("sine_wave curve exists", (
        curve(sine_wave, SineWaveProps),
        member(type(sine), SineWaveProps)
    )),

    test("quadratic parabola curve exists", (
        curve(parabola, ParabolaProps),
        member(type(quadratic), ParabolaProps)
    )),

    % Curve evaluation
    test("Evaluate sine_wave(0) = 0", (
        curve_plot_generator:evaluate_curve(sine_wave, 0, SineY),
        abs(SineY) < 0.001
    )),

    test("Evaluate cosine_wave(0) = 1", (
        curve_plot_generator:evaluate_curve(cosine_wave, 0, CosY),
        abs(CosY - 1) < 0.001
    )),

    test("Evaluate parabola(2) = 4", (
        curve_plot_generator:evaluate_curve(parabola, 2, ParabY),
        abs(ParabY - 4) < 0.001
    )),

    test("Evaluate exp_growth(0) = 1", (
        curve_plot_generator:evaluate_curve(exp_growth, 0, ExpY),
        abs(ExpY - 1) < 0.001
    )),

    % Plot specs
    test("trig_demo spec exists", (
        plot_spec(trig_demo, TrigDemoConfig),
        member(curves(_), TrigDemoConfig)
    )),

    test("polynomial_demo spec exists", (
        plot_spec(polynomial_demo, _)
    )),

    % Component generation
    test("Generate curve component", (
        generate_curve_component(trig_demo, CurveCode1),
        atom_length(CurveCode1, CurveLen),
        CurveLen > 2500
    )),

    test("Component imports Chart.js", (
        generate_curve_component(trig_demo, CurveCode2),
        sub_atom(CurveCode2, _, _, _, 'chart.js')
    )),

    test("Component has useState", (
        generate_curve_component(trig_demo, CurveCode3),
        sub_atom(CurveCode3, _, _, _, 'useState')
    )),

    test("Component has datasets", (
        generate_curve_component(trig_demo, CurveCode4),
        sub_atom(CurveCode4, _, _, _, 'datasets')
    )),

    % Chart.js config
    test("Chart config has x scale", (
        curve_plot_generator:generate_chartjs_config(trig_demo, ChartConfig1),
        sub_atom(ChartConfig1, _, _, _, 'x:')
    )),

    test("Chart config has responsive", (
        curve_plot_generator:generate_chartjs_config(trig_demo, ChartConfig2),
        sub_atom(ChartConfig2, _, _, _, 'responsive')
    )),

    % CSS generation
    test("CSS has chartContainer", (
        generate_curve_styles(trig_demo, CurveCss),
        sub_atom(CurveCss, _, _, _, '.chartContainer')
    )).

% ============================================================================
% MATPLOTLIB GENERATOR TESTS
% ============================================================================

run_matplotlib_generator_tests :-
    format('~n--- Matplotlib Generator Tests ---~n'),

    % Curve queries
    test("Has matplotlib curves", (
        all_matplotlib_curves(Curves1),
        length(Curves1, Count1),
        Count1 >= 8
    )),

    test("sine curve exists", (
        matplotlib_curve(sine, SineProps),
        member(type(sine), SineProps)
    )),

    test("gaussian curve exists", (
        matplotlib_curve(gaussian, GaussianProps),
        member(type(gaussian), GaussianProps)
    )),

    % Plot queries
    test("Has matplotlib plots", (
        all_matplotlib_plots(Plots1),
        length(Plots1, PlotCount),
        PlotCount >= 4
    )),

    test("trig_functions plot exists", (
        matplotlib_plot(trig_functions, TrigConfig),
        member(curves(_), TrigConfig)
    )),

    % Code generation
    test("Generate matplotlib code", (
        generate_matplotlib_code(trig_functions, Code1),
        atom_length(Code1, Len1),
        Len1 > 400
    )),

    test("Code imports numpy", (
        generate_matplotlib_code(trig_functions, Code2),
        sub_atom(Code2, _, _, _, 'import numpy')
    )),

    test("Code imports matplotlib", (
        generate_matplotlib_code(trig_functions, Code3),
        sub_atom(Code3, _, _, _, 'import matplotlib')
    )),

    test("Code has plt.subplots", (
        generate_matplotlib_code(trig_functions, Code4),
        sub_atom(Code4, _, _, _, 'plt.subplots')
    )),

    test("Code has np.linspace", (
        generate_matplotlib_code(trig_functions, Code5),
        sub_atom(Code5, _, _, _, 'np.linspace')
    )),

    test("Code has ax.plot", (
        generate_matplotlib_code(trig_functions, Code6),
        sub_atom(Code6, _, _, _, 'ax.plot')
    )),

    % NumPy expression generation
    test("Sine expression uses np.sin", (
        matplotlib_curve(sine, SineProps2),
        matplotlib_generator:generate_numpy_expression(sine, SineProps2, SineExpr),
        sub_atom(SineExpr, _, _, _, 'np.sin')
    )),

    test("Quadratic expression has x**2", (
        matplotlib_curve(quadratic, QuadProps),
        matplotlib_generator:generate_numpy_expression(quadratic, QuadProps, QuadExpr),
        sub_atom(QuadExpr, _, _, _, 'x**2')
    )),

    test("Gaussian expression has exp", (
        matplotlib_curve(gaussian, GaussProps2),
        matplotlib_generator:generate_numpy_expression(gaussian, GaussProps2, GaussExpr),
        sub_atom(GaussExpr, _, _, _, 'np.exp')
    )),

    % Output code
    test("Show output code", (
        matplotlib_generator:generate_output_code(show, ShowCode),
        sub_atom(ShowCode, _, _, _, 'plt.show')
    )),

    test("PNG output uses savefig", (
        matplotlib_generator:generate_output_code(png("test.png"), PngCode),
        sub_atom(PngCode, _, _, _, 'savefig')
    )),

    % Script generation
    test("Script has main block", (
        generate_matplotlib_script(trig_functions, Script1),
        sub_atom(Script1, _, _, _, '__main__')
    )).

% ============================================================================
% TEST HELPERS
% ============================================================================

test(Name, Goal) :-
    (   catch(Goal, _, fail)
    ->  format('  [PASS] ~w~n', [Name]),
        assertz(test_passed)
    ;   format('  [FAIL] ~w~n', [Name]),
        assertz(test_failed)
    ).

assert_contains(Atom, Substring) :-
    sub_atom(Atom, _, _, _, Substring).

assert_greater(A, B) :-
    A > B.

assert_equals(A, B) :-
    A == B.

assert_approx(A, B, Epsilon) :-
    abs(A - B) < Epsilon.

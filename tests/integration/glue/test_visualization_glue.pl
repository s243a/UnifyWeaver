% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Integration Tests for Visualization Glue Modules
%
% Tests: graph_generator, curve_plot_generator, matplotlib_generator,
%        heatmap_generator, treemap_generator, plot3d_generator

:- use_module('../../../src/unifyweaver/glue/graph_generator').
:- use_module('../../../src/unifyweaver/glue/curve_plot_generator').
:- use_module('../../../src/unifyweaver/glue/matplotlib_generator').
:- use_module('../../../src/unifyweaver/glue/layout_generator').
:- use_module('../../../src/unifyweaver/glue/heatmap_generator').
:- use_module('../../../src/unifyweaver/glue/treemap_generator').
:- use_module('../../../src/unifyweaver/glue/plot3d_generator').
:- use_module('../../../src/unifyweaver/glue/math_expr').
:- use_module('../../../src/unifyweaver/glue/responsive_generator').
:- use_module('../../../src/unifyweaver/glue/accessibility_generator').

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

    % Layout Generator Tests
    run_layout_generator_tests,

    % Layout Integration Tests
    run_layout_integration_tests,

    % Subplot Layout Tests
    run_subplot_layout_tests,

    % Control System Tests
    run_control_system_tests,

    % Wiring System Tests
    run_wiring_system_tests,

    % Heatmap Generator Tests
    run_heatmap_generator_tests,

    % Treemap Generator Tests
    run_treemap_generator_tests,

    % 3D Plot Generator Tests
    run_plot3d_generator_tests,

    % Math Expression Tests
    run_math_expr_tests,

    % Responsive Generator Tests
    run_responsive_generator_tests,

    % Accessibility Generator Tests
    run_accessibility_generator_tests,

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
% LAYOUT GENERATOR TESTS
% ============================================================================

run_layout_generator_tests :-
    format('~n--- Layout Generator Tests ---~n'),

    % Default layouts
    test("Has default layouts defined", (
        default_layout(sidebar_content, _, _),
        default_layout(dashboard, _, _)
    )),

    test("sidebar_content has correct structure", (
        default_layout(sidebar_content, grid, Options),
        member(areas(_), Options),
        member(columns(_), Options)
    )),

    % Themes
    test("Dark theme exists", (
        layout_generator:theme(dark, Props),
        member(background(_), Props),
        member(accent(_), Props)
    )),

    test("Light theme exists", (
        layout_generator:theme(light, LightProps),
        member(background(_), LightProps)
    )),

    test("Midnight theme exists", (
        layout_generator:theme(midnight, _)
    )),

    % CSS Generation
    test("Generate grid CSS", (
        layout_generator:declare_layout(test_layout, grid, [
            areas([["a", "b"]]),
            columns(["200px", "1fr"])
        ]),
        generate_layout_css(test_layout, CSS),
        sub_atom(CSS, _, _, _, 'grid'),
        sub_atom(CSS, _, _, _, 'grid-template-areas')
    )),

    test("Generate theme CSS", (
        generate_theme_css(dark, ThemeCSS),
        sub_atom(ThemeCSS, _, _, _, '--background'),
        sub_atom(ThemeCSS, _, _, _, '--accent')
    )),

    % JSX Generation
    test("Generate layout JSX", (
        layout_generator:declare_layout(test_jsx, grid, [
            areas([["main"]])
        ]),
        assertz(place(test_jsx, main, [content])),
        generate_layout_jsx(test_jsx, JSX),
        sub_atom(JSX, _, _, _, 'className')
    )),

    % Cleanup test layouts
    retractall(layout(test_layout, _, _)),
    retractall(layout(test_jsx, _, _)),
    retractall(place(test_jsx, _, _)).

% ============================================================================
% LAYOUT INTEGRATION TESTS
% ============================================================================

run_layout_integration_tests :-
    format('~n--- Layout Integration Tests ---~n'),

    % Graph with layout
    test("Generate graph with sidebar_content layout", (
        generate_graph_with_layout(family_tree, sidebar_content, GraphCode),
        sub_atom(GraphCode, _, _, _, 'sidebar_content'),
        sub_atom(GraphCode, _, _, _, 'React')
    )),

    test("Generate graph with dashboard layout", (
        generate_graph_with_layout(family_tree, dashboard, GraphCode2),
        sub_atom(GraphCode2, _, _, _, 'dashboard')
    )),

    test("Generate graph full styles", (
        generate_graph_full_styles(family_tree, GraphCSS),
        atom_length(GraphCSS, CSSLen),
        CSSLen > 100
    )),

    % Curve with layout
    test("Generate curve with sidebar_content layout", (
        generate_curve_with_layout(trig_demo, sidebar_content, CurveCode),
        sub_atom(CurveCode, _, _, _, 'sidebar_content'),
        sub_atom(CurveCode, _, _, _, 'Chart')
    )),

    test("Generate curve with dashboard layout", (
        generate_curve_with_layout(trig_demo, dashboard, CurveCode2),
        sub_atom(CurveCode2, _, _, _, 'dashboard')
    )),

    test("Generate curve full styles", (
        generate_curve_full_styles(trig_demo, CurveCSS),
        atom_length(CurveCSS, CSSLen2),
        CSSLen2 > 100
    )),

    % Single layout pattern
    test("Generate graph with single layout", (
        generate_graph_with_layout(simple_graph, single, SingleCode),
        sub_atom(SingleCode, _, _, _, 'React')
    )),

    test("Generate curve with single layout", (
        generate_curve_with_layout(polynomial_demo, single, SingleCurve),
        sub_atom(SingleCurve, _, _, _, 'Chart')
    )).

% ============================================================================
% SUBPLOT LAYOUT TESTS
% ============================================================================

run_subplot_layout_tests :-
    format('~n--- Subplot Layout Tests ---~n'),

    % Setup test subplot layout
    test("Declare subplot layout", (
        layout_generator:assertz(subplot_layout(test_subplot, grid, [rows(2), cols(2)])),
        layout_generator:assertz(subplot_content(test_subplot, pos(1,1), [curve(sine), title("Sine")])),
        layout_generator:assertz(subplot_content(test_subplot, pos(1,2), [curve(cosine), title("Cosine")])),
        layout_generator:assertz(subplot_content(test_subplot, pos(2,1), [curve(quadratic), title("Quadratic")])),
        layout_generator:assertz(subplot_content(test_subplot, pos(2,2), [curve(exponential), title("Exponential")])),
        has_subplot_layout(test_subplot)
    )),

    % Subplot dimensions
    test("Get subplot dimensions", (
        layout_generator:get_subplot_dimensions(test_subplot, Rows, Cols),
        Rows =:= 2,
        Cols =:= 2
    )),

    % Subplot positions
    test("Get subplot positions", (
        layout_generator:get_subplot_positions(test_subplot, Positions),
        length(Positions, 4)
    )),

    % CSS generation (web - synthesized)
    test("Generate subplot CSS", (
        generate_subplot_css(test_subplot, CSS),
        sub_atom(CSS, _, _, _, 'display: grid'),
        sub_atom(CSS, _, _, _, 'grid-template-columns')
    )),

    test("Subplot CSS has cell classes", (
        generate_subplot_css(test_subplot, CSS2),
        sub_atom(CSS2, _, _, _, 'cell-1-1'),
        sub_atom(CSS2, _, _, _, 'cell-2-2')
    )),

    % JSX generation (web - synthesized)
    test("Generate subplot JSX", (
        generate_subplot_jsx(test_subplot, JSX),
        sub_atom(JSX, _, _, _, 'React'),
        sub_atom(JSX, _, _, _, 'className')
    )),

    test("Subplot JSX has chart components", (
        generate_subplot_jsx(test_subplot, JSX2),
        sub_atom(JSX2, _, _, _, 'Chart')
    )),

    % Matplotlib generation (native)
    test("Generate subplot matplotlib", (
        generate_subplot_matplotlib(test_subplot, MplCode),
        sub_atom(MplCode, _, _, _, 'plt.subplots'),
        sub_atom(MplCode, _, _, _, 'axes')
    )),

    test("Matplotlib has 2x2 grid", (
        generate_subplot_matplotlib(test_subplot, MplCode2),
        sub_atom(MplCode2, _, _, _, 'subplots(2, 2')
    )),

    test("Matplotlib has tight_layout", (
        generate_subplot_matplotlib(test_subplot, MplCode3),
        sub_atom(MplCode3, _, _, _, 'tight_layout')
    )),

    % Cleanup
    retractall(subplot_layout(test_subplot, _, _)),
    retractall(subplot_content(test_subplot, _, _)).

% ============================================================================
% CONTROL SYSTEM TESTS
% ============================================================================

run_control_system_tests :-
    format('~n--- Control System Tests ---~n'),

    % Control definitions
    test("Has default controls defined", (
        control(amplitude, slider, _),
        control(curve_type, select, _),
        control(show_grid, checkbox, _)
    )),

    test("Control has expected properties", (
        control(amplitude, slider, Props),
        member(min(_), Props),
        member(max(_), Props),
        member(label(_), Props)
    )),

    % Control panels
    test("Has default control panels", (
        control_panel(curve_controls, _),
        control_panel(display_controls, _)
    )),

    test("Control panel contains controls", (
        control_panel(curve_controls, Controls),
        member(amplitude, Controls),
        member(frequency, Controls)
    )),

    % JSX generation - single control
    % NOTE: Each test must use unique variable names since they share scope
    test("Generate slider JSX", (
        generate_control_jsx(amplitude, SliderJSX),
        sub_atom(SliderJSX, _, _, _, 'type="range"'),
        sub_atom(SliderJSX, _, _, _, 'className')
    )),

    test("Generate select JSX", (
        generate_control_jsx(curve_type, SelectJSX),
        sub_atom(SelectJSX, _, _, _, '<select'),
        sub_atom(SelectJSX, _, _, _, '<option')
    )),

    test("Generate checkbox JSX", (
        generate_control_jsx(show_grid, CheckboxJSX),
        sub_atom(CheckboxJSX, _, _, _, 'type="checkbox"'),
        sub_atom(CheckboxJSX, _, _, _, 'checked')
    )),

    test("Generate color picker JSX", (
        generate_control_jsx(line_color, ColorJSX),
        sub_atom(ColorJSX, _, _, _, 'type="color"')
    )),

    % Control panel JSX
    test("Generate control panel JSX", (
        generate_control_panel_jsx(curve_controls, PanelJSX),
        sub_atom(PanelJSX, _, _, _, 'controlPanel'),
        sub_atom(PanelJSX, _, _, _, 'panelTitle')
    )),

    test("Control panel has multiple controls", (
        generate_control_panel_jsx(curve_controls, PanelJSX2),
        sub_atom(PanelJSX2, _, _, _, 'amplitude'),
        sub_atom(PanelJSX2, _, _, _, 'frequency')
    )),

    % State generation
    test("Generate control state", (
        generate_control_state(curve_controls, StateCode),
        sub_atom(StateCode, _, _, _, 'useState'),
        sub_atom(StateCode, _, _, _, 'amplitude')
    )),

    test("State has correct default values", (
        generate_control_state(display_controls, StateCode2),
        sub_atom(StateCode2, _, _, _, 'true'),  % showGrid default
        sub_atom(StateCode2, _, _, _, '#00d4ff')  % lineColor default
    )),

    % CSS generation
    test("Generate control CSS", (
        generate_control_css(curve_controls, CSS),
        sub_atom(CSS, _, _, _, '.controlPanel'),
        sub_atom(CSS, _, _, _, '.slider'),
        sub_atom(CSS, _, _, _, '.select')
    )),

    % Handler generation
    test("Generate control handlers", (
        generate_control_handlers(curve_controls, Handlers),
        sub_atom(Handlers, _, _, _, 'handle'),
        sub_atom(Handlers, _, _, _, 'Change')
    )).

% ============================================================================
% WIRING SYSTEM TESTS
% ============================================================================

run_wiring_system_tests :-
    format('~n--- Wiring System Tests ---~n'),

    % Wiring spec definitions
    test("Has default wiring specs", (
        wiring_spec(curve_visualization, _),
        wiring_spec(display_settings, _)
    )),

    test("Wiring spec has panel reference", (
        wiring_spec(curve_visualization, Options),
        member(panel(curve_controls), Options)
    )),

    test("Wiring spec has mappings", (
        wiring_spec(curve_visualization, Options),
        member(mappings(Mappings), Options),
        member(amplitude -> amplitude, Mappings)
    )),

    % Prop generation
    test("Generate control props", (
        generate_control_props(curve_controls, Props),
        sub_atom(Props, _, _, _, 'amplitude={amplitude}'),
        sub_atom(Props, _, _, _, 'frequency={frequency}')
    )),

    % Type generation
    test("Generate prop types interface", (
        generate_prop_types(curve_controls, Types),
        sub_atom(Types, _, _, _, 'interface ChartProps'),
        sub_atom(Types, _, _, _, 'amplitude: number')
    )),

    test("Type interface includes all control types", (
        generate_prop_types(display_controls, DisplayTypes),
        sub_atom(DisplayTypes, _, _, _, 'showGrid: boolean'),
        sub_atom(DisplayTypes, _, _, _, 'lineColor: string')
    )),

    % Wired component generation
    test("Generate wired component", (
        generate_wired_component(curve_demo, [panel(curve_controls)], WiredCode),
        sub_atom(WiredCode, _, _, _, 'import React'),
        sub_atom(WiredCode, _, _, _, 'useState')
    )),

    test("Wired component has state declarations", (
        generate_wired_component(test_wired, [panel(curve_controls)], Code),
        sub_atom(Code, _, _, _, '[amplitude, setamplitude]')
    )),

    test("Wired component has control panel", (
        generate_wired_component(test_wired2, [panel(curve_controls)], Code2),
        sub_atom(Code2, _, _, _, 'controlPanel'),
        sub_atom(Code2, _, _, _, 'panelTitle')
    )),

    test("Wired component uses sidebar layout by default", (
        generate_wired_component(test_wired3, [panel(curve_controls)], Code3),
        sub_atom(Code3, _, _, _, 'styles.sidebar'),
        sub_atom(Code3, _, _, _, 'styles.main')
    )).

% ============================================================================
% HEATMAP GENERATOR TESTS
% ============================================================================

run_heatmap_generator_tests :-
    format('~n--- Heatmap Generator Tests ---~n'),

    % Heatmap spec
    test("Has default heatmap specs", (
        heatmap_spec(correlation_demo, _),
        heatmap_spec(activity_demo, _)
    )),

    test("Heatmap spec has config", (
        heatmap_spec(correlation_demo, Config),
        member(title(_), Config),
        member(x_labels(_), Config)
    )),

    % Cell queries
    test("Get heatmap cell value", (
        get_heatmap_cell(correlation_demo, 0, 0, Value),
        Value =:= 1.0
    )),

    test("Heatmap dimensions", (
        heatmap_dimensions(correlation_demo, Rows, Cols),
        Rows =:= 3, Cols =:= 3
    )),

    test("Heatmap value range", (
        heatmap_value_range(correlation_demo, Min, Max),
        Min < Max
    )),

    % Component generation
    test("Generate heatmap component", (
        generate_heatmap_component(correlation_demo, HeatmapCode),
        sub_atom(HeatmapCode, _, _, _, 'React'),
        sub_atom(HeatmapCode, _, _, _, 'heatmapContainer')
    )),

    test("Heatmap has color scale", (
        generate_heatmap_component(correlation_demo, HeatmapCode2),
        sub_atom(HeatmapCode2, _, _, _, 'getColor')
    )),

    % Matplotlib generation
    test("Generate heatmap matplotlib", (
        generate_heatmap_matplotlib(correlation_demo, PyCode),
        sub_atom(PyCode, _, _, _, 'seaborn'),
        sub_atom(PyCode, _, _, _, 'heatmap')
    )),

    % CSS generation
    test("Generate heatmap CSS", (
        generate_heatmap_styles(correlation_demo, HeatmapCSS),
        sub_atom(HeatmapCSS, _, _, _, '.heatmapContainer'),
        sub_atom(HeatmapCSS, _, _, _, '.cell')
    )).

% ============================================================================
% TREEMAP GENERATOR TESTS
% ============================================================================

run_treemap_generator_tests :-
    format('~n--- Treemap Generator Tests ---~n'),

    % Treemap spec
    test("Has default treemap specs", (
        treemap_spec(filesystem_demo, _),
        treemap_spec(budget_demo, _)
    )),

    test("Treemap spec has config", (
        treemap_spec(filesystem_demo, Config),
        member(title(_), Config),
        member(root(_), Config)
    )),

    % Node queries
    test("Treemap has nodes", (
        treemap_node(project_root, null, _, _),
        treemap_node(src, project_root, _, _)
    )),

    test("Get treemap children", (
        treemap_children(filesystem_demo, project_root, Children),
        length(Children, Len),
        Len >= 3
    )),

    test("Calculate total value", (
        treemap_total_value(filesystem_demo, Total),
        Total > 0
    )),

    % Component generation
    test("Generate treemap component", (
        generate_treemap_component(filesystem_demo, TreemapCode),
        sub_atom(TreemapCode, _, _, _, 'React'),
        sub_atom(TreemapCode, _, _, _, 'TreemapNode')
    )),

    test("Treemap has layout algorithm", (
        generate_treemap_component(filesystem_demo, TreemapCode2),
        sub_atom(TreemapCode2, _, _, _, 'calculateLayout')
    )),

    % Plotly generation
    test("Generate treemap plotly", (
        generate_treemap_plotly(budget_demo, PyCode),
        sub_atom(PyCode, _, _, _, 'plotly'),
        sub_atom(PyCode, _, _, _, 'treemap')
    )),

    % CSS generation
    test("Generate treemap CSS", (
        generate_treemap_styles(filesystem_demo, TreemapCSS),
        sub_atom(TreemapCSS, _, _, _, '.treemapContainer'),
        sub_atom(TreemapCSS, _, _, _, '.node')
    )).

% ============================================================================
% 3D PLOT GENERATOR TESTS
% ============================================================================

run_plot3d_generator_tests :-
    format('~n--- 3D Plot Generator Tests ---~n'),

    % Surface specs
    test("Has default surface specs", (
        surface3d(wave_surface, _),
        surface3d(paraboloid, _),
        surface3d(saddle, _)
    )),

    test("Surface spec has config", (
        surface3d(wave_surface, Config),
        member(title(_), Config),
        % Updated to use expr() instead of deprecated function()
        member(expr(_), Config)
    )),

    % Scatter specs
    test("Has scatter demo", (
        scatter3d_spec(cluster_demo, _),
        scatter3d_point(cluster_demo, _, _, _, _)
    )),

    % Line specs
    test("Has line demo", (
        line3d_spec(helix, _),
        line3d_point(helix, _, _, _, _)
    )),

    % Type detection
    test("Detect plot types", (
        plot3d_type(wave_surface, surface),
        plot3d_type(cluster_demo, scatter),
        plot3d_type(helix, line)
    )),

    % Surface component generation
    test("Generate surface component", (
        generate_plot3d_component(wave_surface, SurfaceCode),
        sub_atom(SurfaceCode, _, _, _, 'Plot'),
        sub_atom(SurfaceCode, _, _, _, 'surface')
    )),

    % Scatter component generation
    test("Generate scatter component", (
        generate_plot3d_component(cluster_demo, ScatterCode),
        sub_atom(ScatterCode, _, _, _, 'scatter3d'),
        sub_atom(ScatterCode, _, _, _, 'markers')
    )),

    % Line component generation
    test("Generate line component", (
        generate_plot3d_component(helix, LineCode),
        sub_atom(LineCode, _, _, _, 'scatter3d'),
        sub_atom(LineCode, _, _, _, 'lines')
    )),

    % Matplotlib generation
    test("Generate surface matplotlib", (
        generate_plot3d_matplotlib(wave_surface, PyCode),
        sub_atom(PyCode, _, _, _, 'plot_surface'),
        sub_atom(PyCode, _, _, _, 'Axes3D')
    )),

    test("Generate scatter matplotlib", (
        generate_plot3d_matplotlib(cluster_demo, ScatterPyCode),
        sub_atom(ScatterPyCode, _, _, _, 'scatter')
    )),

    % CSS generation
    test("Generate plot3d CSS", (
        generate_plot3d_styles(wave_surface, PlotCSS),
        sub_atom(PlotCSS, _, _, _, '.plotContainer')
    )).

run_math_expr_tests :-
    format('~n--- Math Expression Tests ---~n'),

    % Basic expression translation
    test("Translate sin expression to JS", (
        math_expr:expr_to_js(sin(x), JS1),
        sub_atom(JS1, _, _, _, 'Math.sin')
    )),

    test("Translate complex expression to JS", (
        math_expr:expr_to_js(sin(x) * cos(y), JS2),
        sub_atom(JS2, _, _, _, 'Math.sin'),
        sub_atom(JS2, _, _, _, 'Math.cos')
    )),

    test("Translate power expression to JS", (
        math_expr:expr_to_js(x ^ 2, JS3),
        sub_atom(JS3, _, _, _, 'Math.pow')
    )),

    test("Translate expression to NumPy", (
        math_expr:expr_to_numpy(sin(x) * cos(y), Py1),
        sub_atom(Py1, _, _, _, 'np.sin'),
        sub_atom(Py1, _, _, _, 'np.cos')
    )),

    test("Constants translated correctly", (
        math_expr:expr_to_js(sin(pi), JS4),
        sub_atom(JS4, _, _, _, 'Math.PI')
    )),

    % Surface expression generation
    test("Surface with expression generates JS", (
        generate_plot3d_component(wave_surface, Code1),
        sub_atom(Code1, _, _, _, 'Math.sin')
    )),

    test("Surface expr translates to Python", (
        generate_plot3d_matplotlib(wave_surface, Py2),
        sub_atom(Py2, _, _, _, 'np.sin')
    )),

    % Curve expression evaluation
    test("Curve expr evaluates correctly", (
        curve_plot_generator:evaluate_curve(gaussian, 0, Y1),
        abs(Y1 - 1.0) < 0.001  % exp(0) = 1
    )),

    test("Curve expr evaluates at x=0 (rational)", (
        curve_plot_generator:evaluate_curve(rational, 0, Y2),
        abs(Y2 - 1.0) < 0.001  % 1/(1+0) = 1
    )),

    % Data curve interpolation
    test("Data curve interpolates correctly", (
        curve_plot_generator:evaluate_curve(sampled_data, 1.5, Y3),
        abs(Y3 - 2.5) < 0.1  % Between 1 and 4
    )),

    % Data surface generation
    test("Data surface generates correctly", (
        generate_plot3d_component(data_surface_example, Code2),
        sub_atom(Code2, _, _, _, 'Measured Data Surface')
    )).

% ============================================================================
% RESPONSIVE GENERATOR TESTS
% ============================================================================

run_responsive_generator_tests :-
    format('~n--- Responsive Generator Tests ---~n'),

    % Breakpoint definitions
    test("Breakpoint definitions exist", (
        breakpoint(xs, _),
        breakpoint(md, _),
        breakpoint(xl, _)
    )),

    test("Mobile breakpoint is max-width 767", (
        breakpoint(mobile, max_width(767))
    )),

    test("Desktop breakpoint is min-width 1024", (
        breakpoint(desktop, min_width(1024))
    )),

    % Media query generation
    test("Generate mobile media query", (
        generate_media_query(mobile, Query),
        sub_atom(Query, _, _, _, 'max-width'),
        sub_atom(Query, _, _, _, '767px')
    )),

    test("Generate desktop media query", (
        generate_media_query(desktop, Query2),
        sub_atom(Query2, _, _, _, 'min-width'),
        sub_atom(Query2, _, _, _, '1024px')
    )),

    test("Generate tablet range query", (
        generate_media_query(tablet, Query3),
        sub_atom(Query3, _, _, _, 'min-width'),
        sub_atom(Query3, _, _, _, 'max-width')
    )),

    % Responsive CSS generation
    test("Generate responsive CSS for collapsible_sidebar", (
        generate_responsive_css(collapsible_sidebar, CSS),
        sub_atom(CSS, _, _, _, '@media'),
        sub_atom(CSS, _, _, _, 'display: grid')
    )),

    test("Responsive CSS includes multiple breakpoints", (
        generate_responsive_css(card_grid, CSS2),
        sub_atom(CSS2, _, _, _, 'min-width: 576px'),
        sub_atom(CSS2, _, _, _, 'min-width: 992px')
    )),

    % Container query generation
    test("Generate container CSS", (
        generate_container_css(chart_container, ContainerCSS),
        sub_atom(ContainerCSS, _, _, _, 'container-type')
    )),

    % Breakpoint ordering
    test("Breakpoints are ordered correctly", (
        breakpoint_order(xs, Order1),
        breakpoint_order(xl, Order2),
        Order1 < Order2
    )).

% ============================================================================
% ACCESSIBILITY GENERATOR TESTS
% ============================================================================

run_accessibility_generator_tests :-
    format('~n--- Accessibility Generator Tests ---~n'),

    % ARIA specification tests
    test("ARIA spec for line_chart has role img", (
        aria_spec(line_chart, Attrs),
        member(role(img), Attrs)
    )),

    test("ARIA spec for data_table has role grid", (
        aria_spec(data_table, Attrs2),
        member(role(grid), Attrs2)
    )),

    % ARIA props generation
    test("Generate ARIA props includes role", (
        generate_aria_props(line_chart, Props),
        sub_atom(Props, _, _, _, 'role')
    )),

    test("Generate ARIA props includes aria-label", (
        generate_aria_props(bar_chart, Props2),
        sub_atom(Props2, _, _, _, 'aria-label')
    )),

    % Keyboard handler generation
    test("Keyboard handler has arrow key support", (
        generate_keyboard_handler(data_table, Handler),
        sub_atom(Handler, _, _, _, 'ArrowUp'),
        sub_atom(Handler, _, _, _, 'ArrowDown')
    )),

    test("Keyboard handler prevents default", (
        generate_keyboard_handler(data_table, Handler2),
        sub_atom(Handler2, _, _, _, 'preventDefault')
    )),

    % Focus trap generation
    test("Focus trap has container selector", (
        generate_focus_trap_jsx(modal_dialog, FocusTrap),
        sub_atom(FocusTrap, _, _, _, 'containerSelector')
    )),

    test("Focus trap returns focus on deactivate", (
        generate_focus_trap_jsx(modal_dialog, FocusTrap2),
        sub_atom(FocusTrap2, _, _, _, 'returnFocusOnDeactivate')
    )),

    % Skip links generation
    test("Skip links has correct structure", (
        generate_skip_links_jsx([main, nav], SkipLinks),
        sub_atom(SkipLinks, _, _, _, 'skipLinks'),
        sub_atom(SkipLinks, _, _, _, 'Skip to')
    )),

    % Live region generation
    test("Live region has aria-live", (
        generate_live_region_jsx(chart_updates, LiveRegion),
        sub_atom(LiveRegion, _, _, _, 'aria-live')
    )),

    test("Live region has role status", (
        generate_live_region_jsx(chart_updates, LiveRegion2),
        sub_atom(LiveRegion2, _, _, _, 'role="status"')
    )),

    % Accessibility CSS generation
    test("Accessibility CSS has screen reader only class", (
        generate_accessibility_css(line_chart, CSS),
        sub_atom(CSS, _, _, _, '.srOnly')
    )),

    test("Accessibility CSS has reduced motion support", (
        generate_accessibility_css(bar_chart, CSS2),
        sub_atom(CSS2, _, _, _, 'prefers-reduced-motion')
    )),

    test("Accessibility CSS has focus styles", (
        generate_accessibility_css(pie_chart, CSS3),
        sub_atom(CSS3, _, _, _, 'focus-visible')
    )),

    % ARIA label query
    test("Get ARIA label returns correct value", (
        get_aria_label(line_chart, Label),
        Label = "Line chart visualization"
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

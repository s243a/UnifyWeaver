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
:- use_module('../../../src/unifyweaver/glue/animation_generator').
:- use_module('../../../src/unifyweaver/glue/interaction_generator').
:- use_module('../../../src/unifyweaver/glue/export_generator').
:- use_module('../../../src/unifyweaver/glue/live_preview_generator').
:- use_module('../../../src/unifyweaver/glue/data_binding_generator').
:- use_module('../../../src/unifyweaver/glue/theme_generator').
:- use_module('../../../src/unifyweaver/glue/animation_presets').
:- use_module('../../../src/unifyweaver/glue/template_library').
:- use_module('../../../src/unifyweaver/glue/lazy_loading_generator').
:- use_module('../../../src/unifyweaver/glue/virtual_scroll_generator').
:- use_module('../../../src/unifyweaver/glue/webworker_generator').

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

    % Animation Generator Tests
    run_animation_generator_tests,

    % Interaction Generator Tests
    run_interaction_generator_tests,

    % Export Generator Tests
    run_export_generator_tests,

    % Live Preview Generator Tests
    run_live_preview_generator_tests,

    % Data Binding Generator Tests
    run_data_binding_generator_tests,

    % Theme Generator Tests
    run_theme_generator_tests,

    % Animation Presets Tests
    run_animation_presets_tests,

    % Template Library Tests
    run_template_library_tests,

    % Lazy Loading Generator Tests
    run_lazy_loading_generator_tests,

    % Virtual Scroll Generator Tests
    run_virtual_scroll_generator_tests,

    % WebWorker Generator Tests
    run_webworker_generator_tests,

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
% ANIMATION GENERATOR TESTS
% ============================================================================

run_animation_generator_tests :-
    format('~n--- Animation Generator Tests ---~n'),

    % Animation definitions
    test("Animation fade_in exists", (
        animation(fade_in, Opts),
        member(keyframes(_), Opts)
    )),

    test("Animation has duration", (
        animation(fade_in, Opts2),
        member(duration(_), Opts2)
    )),

    test("Animation has easing", (
        animation(fade_in, Opts3),
        member(easing(_), Opts3)
    )),

    % Easing functions
    test("Easing ease_out exists", (
        easing(ease_out, _)
    )),

    test("Easing ease_out_back has cubic-bezier", (
        easing(ease_out_back, Easing),
        sub_atom(Easing, _, _, _, 'cubic-bezier')
    )),

    % Keyframes generation
    test("Generate keyframes CSS has @keyframes", (
        generate_keyframes_css(fade_in, KeyframesCSS),
        sub_atom(KeyframesCSS, _, _, _, '@keyframes')
    )),

    test("Generate keyframes CSS has opacity", (
        generate_keyframes_css(fade_in, KeyframesCSS2),
        sub_atom(KeyframesCSS2, _, _, _, 'opacity')
    )),

    % Animation class generation
    test("Generate animation class has animation-name", (
        generate_animation_class(fade_in, fade_in, ClassCSS),
        sub_atom(ClassCSS, _, _, _, 'animation-name')
    )),

    test("Generate animation class has duration", (
        generate_animation_class(scale_in, scale_in, ClassCSS2),
        sub_atom(ClassCSS2, _, _, _, 'animation-duration')
    )),

    % Transition generation
    test("Transition hover_lift exists", (
        transition(hover_lift, TransOpts),
        member(on_hover(_), TransOpts)
    )),

    test("Generate transition CSS has hover", (
        generate_transition_css(hover_lift, TransCSS),
        sub_atom(TransCSS, _, _, _, ':hover')
    )),

    test("Generate transition CSS has transition-property", (
        generate_transition_css(color_fade, TransCSS2),
        sub_atom(TransCSS2, _, _, _, 'transition-property')
    )),

    % Animation utilities
    test("Get animation duration", (
        get_animation_duration(fade_in, Duration),
        Duration =:= 300
    )),

    test("Get animation easing", (
        get_animation_easing(fade_in, Easing2),
        sub_atom(Easing2, _, _, _, 'ease')
    )),

    % React hook generation
    test("Generate animation hook has useState", (
        generate_animation_hook(fade_in, Hook),
        sub_atom(Hook, _, _, _, 'useState')
    )),

    test("Generate animation hook has useCallback", (
        generate_animation_hook(scale_in, Hook2),
        sub_atom(Hook2, _, _, _, 'useCallback')
    )),

    % Chart-specific animations
    test("Chart animation draw_line exists", (
        animation(draw_line, DrawOpts),
        member(keyframes(_), DrawOpts)
    )),

    test("Chart animation bar_grow exists", (
        animation(bar_grow, BarOpts),
        member(keyframes(_), BarOpts)
    )).

% ============================================================================
% INTERACTION GENERATOR TESTS
% ============================================================================

run_interaction_generator_tests :-
    format('~n--- Interaction Generator Tests ---~n'),

    % Interaction specifications
    test("Interaction line_chart exists", (
        interaction(line_chart, Events),
        member(on_hover(_), Events)
    )),

    test("Interaction scatter_plot has zoom", (
        interaction(scatter_plot, Events2),
        member(on_scroll(zoom), Events2)
    )),

    test("Interaction network_graph has pan", (
        interaction(network_graph, Events3),
        member(on_background_drag(pan), Events3)
    )),

    % Event handlers generation
    test("Generate event handlers has mouse enter", (
        generate_event_handlers(line_chart, Handlers),
        sub_atom(Handlers, _, _, _, 'handleMouseEnter')
    )),

    test("Generate event handlers has click", (
        generate_event_handlers(bar_chart, Handlers2),
        sub_atom(Handlers2, _, _, _, 'handleClick')
    )),

    % Tooltip generation
    test("Generate tooltip JSX has component", (
        generate_tooltip_jsx(line_chart, TooltipJSX),
        sub_atom(TooltipJSX, _, _, _, 'Tooltip')
    )),

    test("Generate tooltip CSS has class", (
        generate_tooltip_css(default, TooltipCSS),
        sub_atom(TooltipCSS, _, _, _, '.tooltip')
    )),

    % Zoom controls
    test("Generate zoom controls has component", (
        generate_zoom_controls(scatter_plot, ZoomControls),
        sub_atom(ZoomControls, _, _, _, 'ZoomControls')
    )),

    test("Generate zoom controls has buttons", (
        generate_zoom_controls(default, ZoomControls2),
        sub_atom(ZoomControls2, _, _, _, 'onZoomIn')
    )),

    % Pan handler
    test("Generate pan handler has hook", (
        generate_pan_handler(scatter_plot, PanHandler),
        sub_atom(PanHandler, _, _, _, 'usePan')
    )),

    test("Generate pan handler has start handler", (
        generate_pan_handler(default, PanHandler2),
        sub_atom(PanHandler2, _, _, _, 'handlePanStart')
    )),

    % Drag handler
    test("Generate drag handler for 3D rotation", (
        generate_drag_handler(plot3d, DragHandler),
        sub_atom(DragHandler, _, _, _, 'useRotate')
    )),

    test("Generate drag handler for node move", (
        generate_drag_handler(network_graph, DragHandler2),
        sub_atom(DragHandler2, _, _, _, 'useNodeDrag')
    )),

    % Selection handler
    test("Generate selection handler for multi-select", (
        generate_selection_handler(data_table, SelectHandler),
        sub_atom(SelectHandler, _, _, _, 'useMultiSelect')
    )),

    test("Generate selection handler for brush", (
        generate_selection_handler(scatter_plot, BrushHandler),
        sub_atom(BrushHandler, _, _, _, 'useBrushSelection')
    )),

    % Interaction state
    test("Generate interaction state has tooltip", (
        generate_interaction_state(scatter_plot, State),
        sub_atom(State, _, _, _, 'tooltipVisible')
    )),

    test("Generate interaction state has scale", (
        generate_interaction_state(scatter_plot, State2),
        sub_atom(State2, _, _, _, 'scale')
    )),

    % Utility predicates
    test("Has interaction utility works", (
        has_interaction(line_chart, on_hover)
    )),

    test("Get interaction options works", (
        get_interaction_options(scatter_plot, zoom, ZoomOpts),
        member(enabled(true), ZoomOpts)
    )).

% ============================================================================
% EXPORT GENERATOR TESTS
% ============================================================================

run_export_generator_tests :-
    format('~n--- Export Generator Tests ---~n'),

    % Export config existence
    test("Export config default exists", (
        export_config(default, Opts),
        member(formats(_), Opts)
    )),

    test("Export config line_chart exists", (
        export_config(line_chart, Opts2),
        member(formats(Formats), Opts2),
        member(csv, Formats)
    )),

    % Supported formats
    test("SVG is supported format", (
        supported_format(svg)
    )),

    test("PNG is supported format", (
        supported_format(png)
    )),

    test("PDF is supported format", (
        supported_format(pdf)
    )),

    % Format specifications
    test("SVG format has correct MIME", (
        export_format(svg, SVGOpts),
        member(mime_type('image/svg+xml'), SVGOpts)
    )),

    test("PNG format requires canvas", (
        export_format(png, PNGOpts),
        member(requires_canvas(true), PNGOpts)
    )),

    test("PDF format specifies jsPDF", (
        export_format(pdf, PDFOpts),
        member(library(jspdf), PDFOpts)
    )),

    % Export component generation
    test("Generate export component has ExportControls", (
        generate_export_component(line_chart, Component),
        sub_atom(Component, _, _, _, 'ExportControls')
    )),

    test("Generate export component has useExport", (
        generate_export_component(bar_chart, Component2),
        sub_atom(Component2, _, _, _, 'useExport')
    )),

    % Export hook generation
    test("Generate export hook has SVG export", (
        generate_export_hook(scatter_plot, Hook),
        sub_atom(Hook, _, _, _, 'exportToSVG')
    )),

    test("Generate export hook has PNG export", (
        generate_export_hook(scatter_plot, Hook2),
        sub_atom(Hook2, _, _, _, 'exportToPNG')
    )),

    test("Generate export hook has PDF export", (
        generate_export_hook(scatter_plot, Hook3),
        sub_atom(Hook3, _, _, _, 'exportToPDF')
    )),

    % Export menu generation
    test("Generate export menu has dropdown", (
        generate_export_menu(line_chart, Menu),
        sub_atom(Menu, _, _, _, 'exportDropdown')
    )),

    test("Generate export menu has SVG option", (
        generate_export_menu(bar_chart, Menu2),
        sub_atom(Menu2, _, _, _, 'Export as SVG')
    )),

    % Individual export functions
    test("Generate SVG export uses XMLSerializer", (
        generate_svg_export(default, SVGCode),
        sub_atom(SVGCode, _, _, _, 'XMLSerializer')
    )),

    test("Generate PNG export uses canvas", (
        generate_png_export(default, PNGCode),
        sub_atom(PNGCode, _, _, _, 'canvas')
    )),

    test("Generate PDF export uses jsPDF", (
        generate_pdf_export(default, PDFCode),
        sub_atom(PDFCode, _, _, _, 'jsPDF')
    )),

    % CSS generation
    test("Generate export CSS has controls class", (
        generate_export_css(CSS),
        sub_atom(CSS, _, _, _, '.exportControls')
    )),

    test("Generate export CSS has dropdown styles", (
        generate_export_css(CSS2),
        sub_atom(CSS2, _, _, _, '.exportDropdown')
    )),

    % Utility predicates
    test("Get export formats for line_chart", (
        get_export_formats(line_chart, Formats),
        member(csv, Formats)
    )),

    test("Get export formats default fallback", (
        get_export_formats(nonexistent_chart, Formats2),
        member(svg, Formats2)
    )).

% ============================================================================
% LIVE PREVIEW GENERATOR TESTS
% ============================================================================

run_live_preview_generator_tests :-
    format('~n--- Live Preview Generator Tests ---~n'),

    % Dev server config existence
    test("Dev server config default exists", (
        dev_server_config(default, Opts),
        member(port(_), Opts)
    )),

    test("Dev server config has hot_reload", (
        dev_server_config(default, Opts2),
        member(hot_reload(true), Opts2)
    )),

    % Preview config existence
    test("Preview config default exists", (
        preview_config(default, PrevOpts),
        member(layout(_), PrevOpts)
    )),

    test("Preview config chart_preview has controls", (
        preview_config(chart_preview, PrevOpts2),
        member(controls(_), PrevOpts2)
    )),

    % Dev server generation
    test("Generate dev server uses express", (
        generate_dev_server(default, ServerCode),
        sub_atom(ServerCode, _, _, _, 'express')
    )),

    test("Generate dev server has WebSocket", (
        generate_dev_server(default, ServerCode2),
        sub_atom(ServerCode2, _, _, _, 'WebSocketServer')
    )),

    test("Generate dev server has chokidar", (
        generate_dev_server(default, ServerCode3),
        sub_atom(ServerCode3, _, _, _, 'chokidar')
    )),

    % Vite config generation
    test("Generate Vite config has defineConfig", (
        generate_vite_config(default, ViteConfig),
        sub_atom(ViteConfig, _, _, _, 'defineConfig')
    )),

    test("Generate Vite config has HMR", (
        generate_vite_config(default, ViteConfig2),
        sub_atom(ViteConfig2, _, _, _, 'hmr')
    )),

    % Preview app generation
    test("Generate preview app has useHotReload", (
        generate_preview_app(chart_preview, PreviewApp),
        sub_atom(PreviewApp, _, _, _, 'useHotReload')
    )),

    test("Generate preview app has PreviewPanel", (
        generate_preview_app(default, PreviewApp2),
        sub_atom(PreviewApp2, _, _, _, 'PreviewPanel')
    )),

    % Hot reload hook
    test("Generate hot reload hook has WebSocket", (
        generate_hot_reload_hook(HRHook),
        sub_atom(HRHook, _, _, _, 'WebSocket')
    )),

    test("Generate hot reload hook has reconnect", (
        generate_hot_reload_hook(HRHook2),
        sub_atom(HRHook2, _, _, _, 'reconnect')
    )),

    % State sync hook
    test("Generate state sync hook uses sessionStorage", (
        generate_state_sync_hook(SSHook),
        sub_atom(SSHook, _, _, _, 'sessionStorage')
    )),

    % Preview components
    test("Generate preview wrapper has error boundary", (
        generate_preview_wrapper(default, Wrapper),
        sub_atom(Wrapper, _, _, _, 'getDerivedStateFromError')
    )),

    test("Generate preview panel has error handling", (
        generate_preview_panel(default, Panel),
        sub_atom(Panel, _, _, _, 'errorPanel')
    )),

    test("Generate code editor has tab handling", (
        generate_code_editor(default, Editor),
        sub_atom(Editor, _, _, _, 'Tab')
    )),

    % CSS generation
    test("Generate preview CSS has theme support", (
        generate_preview_css(CSS),
        sub_atom(CSS, _, _, _, 'data-theme')
    )),

    test("Generate preview CSS has previewApp class", (
        generate_preview_css(CSS2),
        sub_atom(CSS2, _, _, _, '.previewApp')
    )),

    % Package.json generation
    test("Generate package.json has vite", (
        generate_package_json(test_project, PackageJSON),
        sub_atom(PackageJSON, _, _, _, '"vite"')
    )),

    test("Generate package.json has react", (
        generate_package_json(test_project, PackageJSON2),
        sub_atom(PackageJSON2, _, _, _, '"react"')
    )),

    % Utility predicates
    test("Get preview port default is 3000", (
        get_preview_port(default, Port),
        Port =:= 3000
    )),

    test("Get watch paths includes Prolog files", (
        get_watch_paths(visualization_preview, Paths),
        member('src/unifyweaver/glue/**/*.pl', Paths)
    )).

% ============================================================================
% DATA BINDING GENERATOR TESTS
% ============================================================================

run_data_binding_generator_tests :-
    format('~nData Binding Generator Tests:~n'),

    % Data source queries
    test("Default data source exists", (
        data_binding_generator:data_source(default, _)
    )),

    test("Time series source exists", (
        data_binding_generator:data_source(time_series, _)
    )),

    test("Graph data source exists", (
        data_binding_generator:data_source(graph_data, _)
    )),

    test("Get source fields returns fields", (
        data_binding_generator:get_source_fields(time_series, Fields),
        length(Fields, 3),
        member(timestamp, Fields),
        member(series, Fields),
        member(value, Fields)
    )),

    % Binding queries
    test("Line chart binding exists", (
        data_binding_generator:binding(line_chart, time_series, _)
    )),

    test("Bar chart binding exists", (
        data_binding_generator:binding(bar_chart, aggregated, _)
    )),

    test("Network graph binding exists", (
        data_binding_generator:binding(network_graph, graph_data, _)
    )),

    test("Get binding mapping returns mapping", (
        data_binding_generator:get_binding_mapping(line_chart, Mapping),
        member(x_axis(timestamp), Mapping),
        member(y_axis(value), Mapping)
    )),

    % Two-way binding
    test("Data table has two-way binding", (
        data_binding_generator:is_two_way(data_table)
    )),

    test("Two-way binding has editable fields", (
        data_binding_generator:two_way_binding(data_table, _, Mapping),
        member(editable([value]), Mapping)
    )),

    % Computed sources
    test("Aggregated computed source exists", (
        data_binding_generator:computed_source(aggregated, _)
    )),

    test("Filtered computed source exists", (
        data_binding_generator:computed_source(filtered, _)
    )),

    % Binding hook generation
    test("Generate binding hook produces code", (
        generate_binding_hook(line_chart, Hook),
        atom_length(Hook, L),
        L > 500
    )),

    test("Generate binding hook has useState", (
        generate_binding_hook(line_chart, Hook),
        sub_atom(Hook, _, _, _, 'useState')
    )),

    test("Generate binding hook has useEffect", (
        generate_binding_hook(line_chart, Hook),
        sub_atom(Hook, _, _, _, 'useEffect')
    )),

    test("Generate binding hook has fetch", (
        generate_binding_hook(line_chart, Hook),
        sub_atom(Hook, _, _, _, 'fetch')
    )),

    % Data provider generation
    test("Generate data provider produces code", (
        generate_data_provider(time_series, Provider),
        atom_length(Provider, L),
        L > 500
    )),

    test("Generate data provider has createContext", (
        generate_data_provider(time_series, Provider),
        sub_atom(Provider, _, _, _, 'createContext')
    )),

    test("Generate data provider has updateRecord", (
        generate_data_provider(time_series, Provider),
        sub_atom(Provider, _, _, _, 'updateRecord')
    )),

    % WebSocket sync generation
    test("Generate websocket sync produces code", (
        generate_websocket_sync(time_series, Sync),
        atom_length(Sync, L),
        L > 500
    )),

    test("Generate websocket sync has WebSocket", (
        generate_websocket_sync(time_series, Sync),
        sub_atom(Sync, _, _, _, 'WebSocket')
    )),

    test("Generate websocket sync has reconnect", (
        generate_websocket_sync(time_series, Sync),
        sub_atom(Sync, _, _, _, 'reconnect')
    )),

    test("Generate websocket sync has subscribe", (
        generate_websocket_sync(time_series, Sync),
        sub_atom(Sync, _, _, _, 'subscribe')
    )),

    % Mutation handler generation
    test("Generate mutation handler produces code", (
        generate_mutation_handler(time_series, Handler),
        atom_length(Handler, L),
        L > 500
    )),

    test("Generate mutation handler has create", (
        generate_mutation_handler(time_series, Handler),
        sub_atom(Handler, _, _, _, 'create')
    )),

    test("Generate mutation handler has update", (
        generate_mutation_handler(time_series, Handler),
        sub_atom(Handler, _, _, _, 'update')
    )),

    test("Generate mutation handler has remove", (
        generate_mutation_handler(time_series, Handler),
        sub_atom(Handler, _, _, _, 'remove')
    )),

    % Type generation
    test("Generate binding types produces interface", (
        generate_binding_types(time_series, Types),
        sub_atom(Types, _, _, _, 'interface')
    )),

    test("Generate binding types has timestamp field", (
        generate_binding_types(time_series, Types),
        sub_atom(Types, _, _, _, 'timestamp')
    )),

    % Type inference
    test("Infer user_id as string", (
        data_binding_generator:infer_field_type(user_id, 'string')
    )),

    test("Infer amount as number", (
        data_binding_generator:infer_field_type(amount, 'number')
    )),

    test("Infer timestamp as Date", (
        data_binding_generator:infer_field_type(timestamp, 'Date | string')
    )),

    test("Infer is_active as boolean", (
        data_binding_generator:infer_field_type(is_active, 'boolean')
    )),

    % Context generation
    test("Generate binding context has createContext", (
        generate_binding_context(time_series, Context),
        sub_atom(Context, _, _, _, 'createContext')
    )),

    % Query generation
    test("Generate fetch query has endpoint", (
        generate_fetch_query(time_series, Query),
        sub_atom(Query, _, _, _, '/api/data/time_series')
    )),

    test("Generate subscribe query has subscribe", (
        generate_subscribe_query(time_series, SubQuery),
        sub_atom(SubQuery, _, _, _, 'subscribe')
    )),

    test("Generate update mutation has PATCH", (
        generate_update_mutation(time_series, Mutation),
        sub_atom(Mutation, _, _, _, 'PATCH')
    )),

    % Management predicates
    test("Declare data source works", (
        data_binding_generator:declare_data_source(test_source, [fields([a, b, c])]),
        data_binding_generator:data_source(test_source, Opts),
        member(fields([a, b, c]), Opts)
    )),

    test("Declare binding works", (
        data_binding_generator:declare_binding(test_component, test_source, [x_axis(a)]),
        data_binding_generator:binding(test_component, test_source, Mapping),
        member(x_axis(a), Mapping)
    )).

% ============================================================================
% THEME GENERATOR TESTS
% ============================================================================

run_theme_generator_tests :-
    format('~n--- Theme Generator Tests ---~n'),

    % Theme existence
    test("Light theme exists", (
        theme_generator:theme(light, Opts),
        member(colors(_), Opts)
    )),

    test("Dark theme exists", (
        theme_generator:theme(dark, Opts),
        member(colors(_), Opts)
    )),

    test("High contrast theme exists", (
        theme_generator:theme(high_contrast, _)
    )),

    test("Corporate theme extends light", (
        theme_generator:theme(corporate, Opts),
        member(extends(light), Opts)
    )),

    % Color palette tests
    test("Blue color palette exists", (
        theme_generator:color_palette(blue, Colors),
        member(c500(_), Colors)
    )),

    test("Slate color palette exists", (
        theme_generator:color_palette(slate, _)
    )),

    test("Emerald color palette exists", (
        theme_generator:color_palette(emerald, _)
    )),

    % Theme resolution
    test("Resolve light theme has colors", (
        theme_generator:resolve_theme(light, Resolved),
        member(colors(_), Resolved)
    )),

    test("Resolve corporate inherits from light", (
        theme_generator:resolve_theme(corporate, Resolved),
        member(typography(_), Resolved)
    )),

    % Get theme utilities
    test("Get theme colors returns list", (
        theme_generator:get_theme_colors(light, Colors),
        member(primary(_), Colors)
    )),

    test("Get theme typography returns settings", (
        theme_generator:get_theme_typography(light, Typo),
        member(font_family(_), Typo)
    )),

    test("Get theme spacing returns values", (
        theme_generator:get_theme_spacing(light, Spacing),
        member(md(_), Spacing)
    )),

    % CSS generation
    test("Generate theme CSS produces output", (
        theme_generator:generate_theme_css(light, CSS),
        atom_length(CSS, L),
        L > 500
    )),

    test("Theme CSS has data-theme attribute", (
        theme_generator:generate_theme_css(light, CSS),
        sub_atom(CSS, _, _, _, 'data-theme="light"')
    )),

    test("Theme CSS has color variables", (
        theme_generator:generate_theme_css(light, CSS),
        sub_atom(CSS, _, _, _, '--color-primary')
    )),

    test("Theme CSS has typography variables", (
        theme_generator:generate_theme_css(dark, CSS),
        sub_atom(CSS, _, _, _, '--typography')
    )),

    test("Theme CSS has spacing variables", (
        theme_generator:generate_theme_css(light, CSS),
        sub_atom(CSS, _, _, _, '--spacing')
    )),

    % Provider generation
    test("Generate theme provider produces React code", (
        theme_generator:generate_theme_provider([light, dark], Provider),
        atom_length(Provider, L),
        L > 1000
    )),

    test("Theme provider has ThemeContext", (
        theme_generator:generate_theme_provider([light, dark], Provider),
        sub_atom(Provider, _, _, _, 'ThemeContext')
    )),

    test("Theme provider has useTheme hook", (
        theme_generator:generate_theme_provider([light, dark], Provider),
        sub_atom(Provider, _, _, _, 'useTheme')
    )),

    % Hook generation
    test("Generate theme hook produces code", (
        theme_generator:generate_theme_hook(Hook),
        atom_length(Hook, L),
        L > 500
    )),

    test("Theme hook has useState", (
        theme_generator:generate_theme_hook(Hook),
        sub_atom(Hook, _, _, _, 'useState')
    )),

    test("Theme hook has toggleTheme", (
        theme_generator:generate_theme_hook(Hook),
        sub_atom(Hook, _, _, _, 'toggleTheme')
    )),

    % Context generation
    test("Generate theme context produces types", (
        theme_generator:generate_theme_context(Context),
        sub_atom(Context, _, _, _, 'interface')
    )),

    test("Theme context has ThemeColors", (
        theme_generator:generate_theme_context(Context),
        sub_atom(Context, _, _, _, 'ThemeColors')
    )),

    % Toggle generation
    test("Generate theme toggle produces component", (
        theme_generator:generate_theme_toggle([light, dark], Toggle),
        sub_atom(Toggle, _, _, _, 'ThemeToggle')
    )),

    test("Theme toggle has select element", (
        theme_generator:generate_theme_toggle([light, dark], Toggle),
        sub_atom(Toggle, _, _, _, '<select')
    )),

    % Type generation
    test("Generate theme types produces TypeScript", (
        theme_generator:generate_theme_types([light, dark], Types),
        sub_atom(Types, _, _, _, 'ThemeName')
    )),

    test("Theme types has ThemeColors interface", (
        theme_generator:generate_theme_types([light, dark], Types),
        sub_atom(Types, _, _, _, 'interface ThemeColors')
    )),

    % Font scale tests
    test("Default font scale exists", (
        theme_generator:font_scale(default, Scale),
        member(base(_), Scale)
    )),

    test("Compact font scale exists", (
        theme_generator:font_scale(compact, _)
    )),

    % Spacing scale tests
    test("Default spacing scale exists", (
        theme_generator:spacing_scale(default, _)
    )).

% ============================================================================
% ANIMATION PRESETS TESTS
% ============================================================================

run_animation_presets_tests :-
    format('~n--- Animation Presets Tests ---~n'),

    % Preset existence
    test("Fade in preset exists", (
        animation_presets:preset(fade_in, Def),
        member(keyframes(_), Def)
    )),

    test("Fade in up preset exists", (
        animation_presets:preset(fade_in_up, _)
    )),

    test("Slide in presets exist", (
        animation_presets:preset(slide_in_up, _),
        animation_presets:preset(slide_in_down, _),
        animation_presets:preset(slide_in_left, _),
        animation_presets:preset(slide_in_right, _)
    )),

    test("Scale in preset exists", (
        animation_presets:preset(scale_in, _)
    )),

    test("Bounce in preset exists", (
        animation_presets:preset(bounce_in, _)
    )),

    % Exit presets
    test("Fade out preset exists", (
        animation_presets:preset(fade_out, Def),
        member(keyframes(_), Def)
    )),

    test("Scale out preset exists", (
        animation_presets:preset(scale_out, _)
    )),

    % Attention presets
    test("Pulse preset exists", (
        animation_presets:preset(pulse, Def),
        member(iteration_count(infinite), Def)
    )),

    test("Bounce preset exists", (
        animation_presets:preset(bounce, _)
    )),

    test("Shake preset exists", (
        animation_presets:preset(shake, _)
    )),

    test("Heartbeat preset exists", (
        animation_presets:preset(heartbeat, _)
    )),

    test("Jello preset exists", (
        animation_presets:preset(jello, _)
    )),

    % Chart presets
    test("Chart draw preset exists", (
        animation_presets:preset(chart_draw, _)
    )),

    test("Bar grow preset exists", (
        animation_presets:preset(bar_grow, _)
    )),

    test("Pie reveal preset exists", (
        animation_presets:preset(pie_reveal, _)
    )),

    test("Data point pop preset exists", (
        animation_presets:preset(data_point_pop, _)
    )),

    % Category tests
    test("Fade in is entry category", (
        animation_presets:preset_category(fade_in, entry)
    )),

    test("Fade out is exit category", (
        animation_presets:preset_category(fade_out, exit)
    )),

    test("Pulse is attention category", (
        animation_presets:preset_category(pulse, attention)
    )),

    test("Chart draw is chart category", (
        animation_presets:preset_category(chart_draw, chart)
    )),

    % List presets
    test("List presets returns multiple", (
        animation_presets:list_presets(Presets),
        length(Presets, L),
        L > 20
    )),

    test("List presets by category works", (
        animation_presets:list_presets_by_category(entry, EntryPresets),
        length(EntryPresets, L),
        L > 5
    )),

    % CSS generation
    test("Generate preset CSS produces output", (
        animation_presets:generate_preset_css(fade_in, CSS),
        atom_length(CSS, L),
        L > 100
    )),

    test("Preset CSS has @keyframes", (
        animation_presets:generate_preset_css(fade_in, CSS),
        sub_atom(CSS, _, _, _, '@keyframes')
    )),

    test("Preset CSS has animation class", (
        animation_presets:generate_preset_css(fade_in, CSS),
        sub_atom(CSS, _, _, _, '.animate-fade_in')
    )),

    test("Preset CSS has animation-duration", (
        animation_presets:generate_preset_css(scale_in, CSS),
        sub_atom(CSS, _, _, _, 'animation-duration')
    )),

    % Keyframes generation
    test("Generate preset keyframes", (
        animation_presets:generate_preset_keyframes(fade_in, Keyframes),
        sub_atom(Keyframes, _, _, _, '@keyframes fade_in')
    )),

    % All presets CSS
    test("Generate all presets CSS", (
        animation_presets:generate_all_presets_css(AllCSS),
        atom_length(AllCSS, L),
        L > 3000
    )),

    % Hook generation
    test("Generate preset hook produces code", (
        animation_presets:generate_preset_hook(Hook),
        atom_length(Hook, L),
        L > 1000
    )),

    test("Preset hook has useAnimation", (
        animation_presets:generate_preset_hook(Hook),
        sub_atom(Hook, _, _, _, 'useAnimation')
    )),

    test("Preset hook has useStaggeredAnimation", (
        animation_presets:generate_preset_hook(Hook),
        sub_atom(Hook, _, _, _, 'useStaggeredAnimation')
    )),

    test("Preset hook has AnimationPreset type", (
        animation_presets:generate_preset_hook(Hook),
        sub_atom(Hook, _, _, _, 'AnimationPreset')
    )),

    % Component generation
    test("Generate preset component produces code", (
        animation_presets:generate_preset_component(fade_in, Component),
        atom_length(Component, L),
        L > 500
    )),

    test("Preset component has forwardRef", (
        animation_presets:generate_preset_component(fade_in, Component),
        sub_atom(Component, _, _, _, 'forwardRef')
    )),

    % Utility predicates
    test("Get preset duration", (
        animation_presets:preset_duration(fade_in, Duration),
        Duration =:= 300
    )),

    test("Get preset easing", (
        animation_presets:preset_easing(fade_in, Easing),
        sub_atom(Easing, _, _, _, 'ease')
    )),

    % Composition
    test("Compose presets combines definitions", (
        animation_presets:compose_presets([fade_in, scale_in], [], Combined),
        member(keyframes(_), Combined)
    )),

    test("Sequence presets creates sequence", (
        animation_presets:sequence_presets([fade_in, scale_in], [stagger(100)], Sequence),
        length(Sequence, 2)
    )),

    % Custom class generation
    test("Generate preset class with options", (
        animation_presets:generate_preset_class(fade_in, [duration(500)], Class),
        sub_atom(Class, _, _, _, '500ms')
    )).

% ============================================================================
% TEMPLATE LIBRARY TESTS
% ============================================================================

run_template_library_tests :-
    format('~n--- Template Library Tests ---~n'),

    % Template types
    test("Dashboard template type exists", (
        template_library:template_type(dashboard, _)
    )),

    test("Report template type exists", (
        template_library:template_type(report, _)
    )),

    test("Explorer template type exists", (
        template_library:template_type(explorer, _)
    )),

    test("Presentation template type exists", (
        template_library:template_type(presentation, _)
    )),

    % Dashboard templates
    test("Analytics dashboard exists", (
        template_library:template(analytics_dashboard, dashboard, Opts),
        member(widgets(_), Opts)
    )),

    test("Sales dashboard exists", (
        template_library:template(sales_dashboard, dashboard, _)
    )),

    test("Realtime monitor exists", (
        template_library:template(realtime_monitor, dashboard, _)
    )),

    % Report templates
    test("Monthly report exists", (
        template_library:template(monthly_report, report, Opts),
        member(sections(_), Opts)
    )),

    test("Comparison report exists", (
        template_library:template(comparison_report, report, _)
    )),

    % Explorer templates
    test("Data explorer exists", (
        template_library:template(data_explorer, explorer, _)
    )),

    test("Chart explorer exists", (
        template_library:template(chart_explorer, explorer, _)
    )),

    % Presentation templates
    test("Slide deck exists", (
        template_library:template(slide_deck, presentation, _)
    )),

    % Template queries
    test("List templates returns multiple", (
        template_library:list_templates(Templates),
        length(Templates, L),
        L >= 7
    )),

    test("List templates by type works", (
        template_library:list_templates_by_type(dashboard, Dashboards),
        length(Dashboards, L),
        L >= 3
    )),

    test("Get template returns spec", (
        template_library:get_template(analytics_dashboard, spec(analytics_dashboard, dashboard, _))
    )),

    % Feature checks
    test("Analytics dashboard has date_range_picker", (
        template_library:template_has_feature(analytics_dashboard, date_range_picker)
    )),

    test("Analytics dashboard has export_pdf", (
        template_library:template_has_feature(analytics_dashboard, export_pdf)
    )),

    % Chart extraction
    test("Get template charts returns chart types", (
        template_library:get_template_charts(analytics_dashboard, Charts),
        member(line_chart, Charts)
    )),

    % JSX generation
    test("Generate template JSX produces code", (
        template_library:generate_template_jsx(analytics_dashboard, JSX),
        atom_length(JSX, L),
        L > 500
    )),

    test("Template JSX has React import", (
        template_library:generate_template_jsx(analytics_dashboard, JSX),
        sub_atom(JSX, _, _, _, 'import React')
    )),

    test("Template JSX has component interface", (
        template_library:generate_template_jsx(analytics_dashboard, JSX),
        sub_atom(JSX, _, _, _, 'interface')
    )),

    test("Template JSX has template-header", (
        template_library:generate_template_jsx(analytics_dashboard, JSX),
        sub_atom(JSX, _, _, _, 'template-header')
    )),

    % CSS generation
    test("Generate template CSS produces output", (
        template_library:generate_template_css(analytics_dashboard, CSS),
        atom_length(CSS, L),
        L > 500
    )),

    test("Template CSS has template class", (
        template_library:generate_template_css(analytics_dashboard, CSS),
        sub_atom(CSS, _, _, _, '.template-analytics_dashboard')
    )),

    test("Template CSS has dashboard-grid", (
        template_library:generate_template_css(analytics_dashboard, CSS),
        sub_atom(CSS, _, _, _, '.dashboard-grid')
    )),

    test("Template CSS has widget styles", (
        template_library:generate_template_css(analytics_dashboard, CSS),
        sub_atom(CSS, _, _, _, '.widget')
    )),

    test("Template CSS has responsive styles", (
        template_library:generate_template_css(analytics_dashboard, CSS),
        sub_atom(CSS, _, _, _, '@media')
    )),

    % Report CSS generation
    test("Report template has print styles support", (
        template_library:generate_template_css(monthly_report, CSS),
        sub_atom(CSS, _, _, _, '.report')
    )),

    % Print styles
    test("Generate print styles produces output", (
        template_library:generate_print_styles(PrintCSS),
        sub_atom(PrintCSS, _, _, _, '@media print')
    )),

    test("Print styles has page break", (
        template_library:generate_print_styles(PrintCSS),
        sub_atom(PrintCSS, _, _, _, 'page-break')
    )),

    % Types generation
    test("Generate template types produces interface", (
        template_library:generate_template_types(analytics_dashboard, Types),
        sub_atom(Types, _, _, _, 'interface')
    )),

    test("Template types has Props interface", (
        template_library:generate_template_types(analytics_dashboard, Types),
        sub_atom(Types, _, _, _, 'Props')
    )),

    % Hook generation
    test("Generate template hook produces code", (
        template_library:generate_template_hook(analytics_dashboard, Hook),
        atom_length(Hook, L),
        L > 500
    )),

    test("Template hook has useState", (
        template_library:generate_template_hook(analytics_dashboard, Hook),
        sub_atom(Hook, _, _, _, 'useState')
    )),

    test("Template hook has refresh function", (
        template_library:generate_template_hook(analytics_dashboard, Hook),
        sub_atom(Hook, _, _, _, 'refresh')
    )),

    % Dashboard layout
    test("Generate dashboard layout produces CSS", (
        template_library:template(analytics_dashboard, _, Opts),
        template_library:generate_dashboard_layout(Opts, Layout),
        sub_atom(Layout, _, _, _, 'grid-template-areas')
    )),

    % Dashboard widgets
    test("Generate dashboard widgets returns list", (
        template_library:template(analytics_dashboard, _, Opts),
        template_library:generate_dashboard_widgets(Opts, Widgets),
        length(Widgets, L),
        L >= 5
    )),

    % Report layout
    test("Generate report layout produces page config", (
        template_library:template(monthly_report, _, Opts),
        template_library:generate_report_layout(Opts, Layout),
        sub_atom(Layout, _, _, _, '@page')
    )),

    % Complete template generation
    test("Generate template produces complete bundle", (
        template_library:generate_template(analytics_dashboard, Code),
        atom_length(Code, L),
        L > 2000
    )),

    test("Complete template has Types section", (
        template_library:generate_template(analytics_dashboard, Code),
        sub_atom(Code, _, _, _, '--- Types ---')
    )),

    test("Complete template has Hook section", (
        template_library:generate_template(analytics_dashboard, Code),
        sub_atom(Code, _, _, _, '--- Hook ---')
    )),

    test("Complete template has Component section", (
        template_library:generate_template(analytics_dashboard, Code),
        sub_atom(Code, _, _, _, '--- Component ---')
    )),

    test("Complete template has Styles section", (
        template_library:generate_template(analytics_dashboard, Code),
        sub_atom(Code, _, _, _, '--- Styles ---')
    )).

% ============================================================================
% LAZY LOADING GENERATOR TESTS
% ============================================================================

run_lazy_loading_generator_tests :-
    format('~n--- Lazy Loading Generator Tests ---~n'),

    % Config existence
    test("Default lazy config exists", (
        lazy_loading_generator:lazy_config(default, Config),
        member(strategy(_), Config)
    )),

    test("Infinite scroll config exists", (
        lazy_loading_generator:lazy_config(infinite_scroll, _)
    )),

    test("Windowed config exists", (
        lazy_loading_generator:lazy_config(windowed, _)
    )),

    test("Chunked config exists", (
        lazy_loading_generator:lazy_config(chunked, _)
    )),

    % Strategy tests
    test("Default strategy is pagination", (
        lazy_loading_generator:lazy_strategy(default, pagination)
    )),

    test("Infinite scroll strategy", (
        lazy_loading_generator:lazy_strategy(infinite_scroll, infinite)
    )),

    % Hook generation
    test("Generate lazy hook produces code", (
        lazy_loading_generator:generate_lazy_hook(default, Hook),
        atom_length(Hook, L),
        L > 1000
    )),

    test("Lazy hook has useLazyData", (
        lazy_loading_generator:generate_lazy_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'useLazyData')
    )),

    test("Lazy hook has cache management", (
        lazy_loading_generator:generate_lazy_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'cache')
    )),

    test("Lazy hook has pagination", (
        lazy_loading_generator:generate_lazy_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'loadPage')
    )),

    % Pagination hook generation
    test("Generate pagination hook produces code", (
        lazy_loading_generator:generate_pagination_hook(default, PagHook),
        atom_length(PagHook, L),
        L > 1000
    )),

    test("Pagination hook has usePagination", (
        lazy_loading_generator:generate_pagination_hook(default, PagHook),
        sub_atom(PagHook, _, _, _, 'usePagination')
    )),

    test("Pagination hook has controls component", (
        lazy_loading_generator:generate_pagination_hook(default, PagHook),
        sub_atom(PagHook, _, _, _, 'PaginationControls')
    )),

    test("Pagination hook has page navigation", (
        lazy_loading_generator:generate_pagination_hook(default, PagHook),
        sub_atom(PagHook, _, _, _, 'nextPage')
    )),

    % Infinite scroll generation
    test("Generate infinite scroll produces code", (
        lazy_loading_generator:generate_infinite_scroll(infinite_scroll, InfHook),
        atom_length(InfHook, L),
        L > 1000
    )),

    test("Infinite scroll has intersection observer", (
        lazy_loading_generator:generate_infinite_scroll(infinite_scroll, InfHook),
        sub_atom(InfHook, _, _, _, 'IntersectionObserver')
    )),

    test("Infinite scroll has sentinel ref", (
        lazy_loading_generator:generate_infinite_scroll(infinite_scroll, InfHook),
        sub_atom(InfHook, _, _, _, 'sentinelRef')
    )),

    test("Infinite scroll has container component", (
        lazy_loading_generator:generate_infinite_scroll(infinite_scroll, InfHook),
        sub_atom(InfHook, _, _, _, 'InfiniteScrollContainer')
    )),

    % Lazy loader generation
    test("Generate lazy loader produces code", (
        lazy_loading_generator:generate_lazy_loader(chunked, Loader),
        atom_length(Loader, L),
        L > 500
    )),

    test("Lazy loader has LazyLoader class", (
        lazy_loading_generator:generate_lazy_loader(chunked, Loader),
        sub_atom(Loader, _, _, _, 'LazyLoader')
    )),

    test("Lazy loader has loadRange method", (
        lazy_loading_generator:generate_lazy_loader(chunked, Loader),
        sub_atom(Loader, _, _, _, 'loadRange')
    )),

    % Lazy component generation
    test("Generate lazy component produces code", (
        lazy_loading_generator:generate_lazy_component(default, Component),
        atom_length(Component, L),
        L > 500
    )),

    test("Lazy component has createLazyComponent", (
        lazy_loading_generator:generate_lazy_component(default, Component),
        sub_atom(Component, _, _, _, 'createLazyComponent')
    )),

    test("Lazy component has error boundary", (
        lazy_loading_generator:generate_lazy_component(default, Component),
        sub_atom(Component, _, _, _, 'LazyErrorBoundary')
    )),

    test("Lazy component has Suspense", (
        lazy_loading_generator:generate_lazy_component(default, Component),
        sub_atom(Component, _, _, _, 'Suspense')
    )).

% ============================================================================
% VIRTUAL SCROLL GENERATOR TESTS
% ============================================================================

run_virtual_scroll_generator_tests :-
    format('~n--- Virtual Scroll Generator Tests ---~n'),

    % Config existence
    test("Default virtual config exists", (
        virtual_scroll_generator:virtual_config(default, Config),
        member(item_height(_), Config)
    )),

    test("Compact list config exists", (
        virtual_scroll_generator:virtual_config(compact_list, _)
    )),

    test("Large table config exists", (
        virtual_scroll_generator:virtual_config(large_table, _)
    )),

    test("Card grid config exists", (
        virtual_scroll_generator:virtual_config(card_grid, _)
    )),

    % Item height utility
    test("Get item height returns default", (
        virtual_scroll_generator:get_item_height(default, Height),
        Height =:= 40
    )),

    test("Get item height for large table", (
        virtual_scroll_generator:get_item_height(large_table, Height),
        Height =:= 48
    )),

    % Hook generation
    test("Generate virtual scroll hook produces code", (
        virtual_scroll_generator:generate_virtual_scroll_hook(default, Hook),
        atom_length(Hook, L),
        L > 1000
    )),

    test("Virtual scroll hook has useVirtualScroll", (
        virtual_scroll_generator:generate_virtual_scroll_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'useVirtualScroll')
    )),

    test("Virtual scroll hook has binary search", (
        virtual_scroll_generator:generate_virtual_scroll_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'findStartIndex')
    )),

    test("Virtual scroll hook has scrollToIndex", (
        virtual_scroll_generator:generate_virtual_scroll_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'scrollToIndex')
    )),

    % Virtual list generation
    test("Generate virtual list produces code", (
        virtual_scroll_generator:generate_virtual_list(default, List),
        atom_length(List, L),
        L > 500
    )),

    test("Virtual list has VirtualList component", (
        virtual_scroll_generator:generate_virtual_list(default, List),
        sub_atom(List, _, _, _, 'VirtualList')
    )),

    test("Virtual list has renderItem prop", (
        virtual_scroll_generator:generate_virtual_list(default, List),
        sub_atom(List, _, _, _, 'renderItem')
    )),

    test("Virtual list has virtualItems mapping", (
        virtual_scroll_generator:generate_virtual_list(default, List),
        sub_atom(List, _, _, _, 'virtualItems.map')
    )),

    % Virtual table generation
    test("Generate virtual table produces code", (
        virtual_scroll_generator:generate_virtual_table(large_table, Table),
        atom_length(Table, L),
        L > 1000
    )),

    test("Virtual table has VirtualTable component", (
        virtual_scroll_generator:generate_virtual_table(large_table, Table),
        sub_atom(Table, _, _, _, 'VirtualTable')
    )),

    test("Virtual table has columns prop", (
        virtual_scroll_generator:generate_virtual_table(large_table, Table),
        sub_atom(Table, _, _, _, 'columns: Column')
    )),

    test("Virtual table has sticky header support", (
        virtual_scroll_generator:generate_virtual_table(large_table, Table),
        sub_atom(Table, _, _, _, 'stickyHeader')
    )),

    test("Virtual table has sorting support", (
        virtual_scroll_generator:generate_virtual_table(large_table, Table),
        sub_atom(Table, _, _, _, 'sortColumn')
    )),

    % Virtual grid generation
    test("Generate virtual grid produces code", (
        virtual_scroll_generator:generate_virtual_grid(card_grid, Grid),
        atom_length(Grid, L),
        L > 500
    )),

    test("Virtual grid has VirtualGrid component", (
        virtual_scroll_generator:generate_virtual_grid(card_grid, Grid),
        sub_atom(Grid, _, _, _, 'VirtualGrid')
    )),

    test("Virtual grid has columnsCount calculation", (
        virtual_scroll_generator:generate_virtual_grid(card_grid, Grid),
        sub_atom(Grid, _, _, _, 'columnsCount')
    )),

    % CSS generation
    test("Generate virtual scroll CSS produces output", (
        virtual_scroll_generator:generate_virtual_scroll_css(CSS),
        atom_length(CSS, L),
        L > 500
    )),

    test("Virtual scroll CSS has list styles", (
        virtual_scroll_generator:generate_virtual_scroll_css(CSS),
        sub_atom(CSS, _, _, _, '.virtual-list')
    )),

    test("Virtual scroll CSS has table styles", (
        virtual_scroll_generator:generate_virtual_scroll_css(CSS),
        sub_atom(CSS, _, _, _, '.virtual-table')
    )),

    test("Virtual scroll CSS has grid styles", (
        virtual_scroll_generator:generate_virtual_scroll_css(CSS),
        sub_atom(CSS, _, _, _, '.virtual-grid')
    )),

    test("Virtual scroll CSS has performance optimizations", (
        virtual_scroll_generator:generate_virtual_scroll_css(CSS),
        sub_atom(CSS, _, _, _, 'contain: strict')
    )).

% ============================================================================
% WEBWORKER GENERATOR TESTS
% ============================================================================

run_webworker_generator_tests :-
    format('~n--- WebWorker Generator Tests ---~n'),

    % Config existence
    test("Default worker config exists", (
        webworker_generator:worker_config(default, Config),
        member(operations(_), Config)
    )),

    test("Data processor config exists", (
        webworker_generator:worker_config(data_processor, _)
    )),

    test("Chart calculator config exists", (
        webworker_generator:worker_config(chart_calculator, _)
    )),

    test("Statistics config exists", (
        webworker_generator:worker_config(statistics, _)
    )),

    % Operations utility
    test("Get worker operations for data processor", (
        webworker_generator:worker_operations(data_processor, Ops),
        member(sort, Ops),
        member(filter, Ops),
        member(aggregate, Ops)
    )),

    test("Get worker operations for statistics", (
        webworker_generator:worker_operations(statistics, Ops),
        member(mean, Ops),
        member(stddev, Ops)
    )),

    % Worker generation
    test("Generate worker produces code", (
        webworker_generator:generate_worker(data_processor, Worker),
        atom_length(Worker, L),
        L > 1000
    )),

    test("Worker has sort operation", (
        webworker_generator:generate_worker(data_processor, Worker),
        sub_atom(Worker, _, _, _, 'sort:')
    )),

    test("Worker has filter operation", (
        webworker_generator:generate_worker(data_processor, Worker),
        sub_atom(Worker, _, _, _, 'filter:')
    )),

    test("Worker has aggregate operation", (
        webworker_generator:generate_worker(data_processor, Worker),
        sub_atom(Worker, _, _, _, 'aggregate:')
    )),

    test("Worker has message handler", (
        webworker_generator:generate_worker(data_processor, Worker),
        sub_atom(Worker, _, _, _, 'self.onmessage')
    )),

    test("Worker has performance timing", (
        webworker_generator:generate_worker(data_processor, Worker),
        sub_atom(Worker, _, _, _, 'performance.now()')
    )),

    % Chart worker generation
    test("Generate chart worker produces code", (
        webworker_generator:generate_chart_worker(ChartWorker),
        atom_length(ChartWorker, L),
        L > 500
    )),

    test("Chart worker has interpolate", (
        webworker_generator:generate_chart_worker(ChartWorker),
        sub_atom(ChartWorker, _, _, _, 'interpolate')
    )),

    test("Chart worker has downsample", (
        webworker_generator:generate_chart_worker(ChartWorker),
        sub_atom(ChartWorker, _, _, _, 'downsample')
    )),

    % Hook generation
    test("Generate worker hook produces code", (
        webworker_generator:generate_worker_hook(default, Hook),
        atom_length(Hook, L),
        L > 1000
    )),

    test("Worker hook has useWorker", (
        webworker_generator:generate_worker_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'useWorker')
    )),

    test("Worker hook has execute method", (
        webworker_generator:generate_worker_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'execute')
    )),

    test("Worker hook has timeout handling", (
        webworker_generator:generate_worker_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'timeout')
    )),

    test("Worker hook has terminate method", (
        webworker_generator:generate_worker_hook(default, Hook),
        sub_atom(Hook, _, _, _, 'terminate')
    )),

    % Pool generation
    test("Generate worker pool produces code", (
        webworker_generator:generate_worker_pool(default, Pool),
        atom_length(Pool, L),
        L > 500
    )),

    test("Worker pool has WorkerPool class", (
        webworker_generator:generate_worker_pool(default, Pool),
        sub_atom(Pool, _, _, _, 'WorkerPool')
    )),

    test("Worker pool has execute method", (
        webworker_generator:generate_worker_pool(default, Pool),
        sub_atom(Pool, _, _, _, 'execute(')
    )),

    test("Worker pool has task queue", (
        webworker_generator:generate_worker_pool(default, Pool),
        sub_atom(Pool, _, _, _, 'taskQueue')
    )),

    test("Worker pool has idle timeout", (
        webworker_generator:generate_worker_pool(default, Pool),
        sub_atom(Pool, _, _, _, 'idleTimeout')
    )),

    % Data processor worker generation
    test("Generate data processor worker produces code", (
        webworker_generator:generate_data_processor_worker(DPWorker),
        atom_length(DPWorker, L),
        L > 1000
    )),

    test("Data processor worker has paginate", (
        webworker_generator:generate_data_processor_worker(DPWorker),
        sub_atom(DPWorker, _, _, _, 'paginate')
    )).

% ============================================================================
% TEST HELPERS
% ============================================================================

test(Name, Goal) :-
    % Use copy_term to ensure fresh variables for each test
    copy_term(Goal, FreshGoal),
    (   catch(FreshGoal, _, fail)
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

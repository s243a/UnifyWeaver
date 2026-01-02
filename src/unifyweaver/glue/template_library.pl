% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Template Library - Pre-built Visualization Templates
%
% This module provides pre-built templates for common visualization
% patterns like dashboards, reports, and data explorers.
%
% Usage:
%   % Use a dashboard template
%   template(my_dashboard, analytics_dashboard, [
%       title("Sales Analytics"),
%       charts([line_chart, bar_chart, pie_chart])
%   ]).
%
%   % Generate template code
%   ?- generate_template(my_dashboard, Code).

:- module(template_library, [
    % Template specifications
    template/3,                     % template(+Name, +Type, +Options)
    template_type/2,                % template_type(+Type, +Description)

    % Template queries
    get_template/2,                 % get_template(+Name, -Spec)
    list_templates/1,               % list_templates(-Templates)
    list_templates_by_type/2,       % list_templates_by_type(+Type, -Templates)

    % Generation predicates
    generate_template/2,            % generate_template(+Name, -Code)
    generate_template_jsx/2,        % generate_template_jsx(+Name, -JSX)
    generate_template_css/2,        % generate_template_css(+Name, -CSS)
    generate_template_types/2,      % generate_template_types(+Name, -Types)
    generate_template_hook/2,       % generate_template_hook(+Name, -Hook)

    % Dashboard templates
    generate_dashboard_layout/2,    % generate_dashboard_layout(+Spec, -Layout)
    generate_dashboard_widgets/2,   % generate_dashboard_widgets(+Spec, -Widgets)

    % Report templates
    generate_report_layout/2,       % generate_report_layout(+Spec, -Layout)
    generate_print_styles/1,        % generate_print_styles(-CSS)

    % Utility predicates
    template_has_feature/2,         % template_has_feature(+Name, +Feature)
    get_template_charts/2,          % get_template_charts(+Name, -Charts)

    % Management
    declare_template/3,             % declare_template(+Name, +Type, +Options)
    clear_templates/0,              % clear_templates

    % Testing
    test_template_library/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic template/3.
:- dynamic template_type/2.

:- discontiguous template/3.

% ============================================================================
% TEMPLATE TYPES
% ============================================================================

template_type(dashboard, "Multi-widget dashboard with configurable layout").
template_type(report, "Print-optimized report with charts and tables").
template_type(explorer, "Interactive data exploration interface").
template_type(presentation, "Slide-based visualization presentation").
template_type(monitor, "Real-time monitoring dashboard").
template_type(comparison, "Side-by-side comparison layout").

% ============================================================================
% DASHBOARD TEMPLATES
% ============================================================================

template(analytics_dashboard, dashboard, [
    title("Analytics Dashboard"),
    description("Overview of key metrics and trends"),
    layout(grid),
    grid_config([
        columns(4),
        gap('1rem'),
        areas([
            ["kpi1", "kpi2", "kpi3", "kpi4"],
            ["main_chart", "main_chart", "side_chart", "side_chart"],
            ["table", "table", "table", "table"]
        ])
    ]),
    widgets([
        widget(kpi1, kpi_card, [metric(revenue), label("Revenue"), icon(dollar)]),
        widget(kpi2, kpi_card, [metric(users), label("Users"), icon(users)]),
        widget(kpi3, kpi_card, [metric(orders), label("Orders"), icon(cart)]),
        widget(kpi4, kpi_card, [metric(conversion), label("Conversion"), icon(percent)]),
        widget(main_chart, line_chart, [title("Trend Over Time"), height(300)]),
        widget(side_chart, pie_chart, [title("Distribution"), height(300)]),
        widget(table, data_table, [title("Recent Activity"), rows(10)])
    ]),
    features([
        date_range_picker,
        export_pdf,
        refresh_button,
        fullscreen
    ]),
    theme(light),
    responsive([
        breakpoint(mobile, [columns(1)]),
        breakpoint(tablet, [columns(2)]),
        breakpoint(desktop, [columns(4)])
    ])
]).

template(sales_dashboard, dashboard, [
    title("Sales Dashboard"),
    description("Sales performance and pipeline overview"),
    layout(grid),
    grid_config([
        columns(3),
        gap('1rem')
    ]),
    widgets([
        widget(revenue, kpi_card, [metric(total_revenue), trend(true)]),
        widget(deals, kpi_card, [metric(deals_closed), trend(true)]),
        widget(pipeline, kpi_card, [metric(pipeline_value), trend(true)]),
        widget(sales_chart, bar_chart, [title("Monthly Sales"), stacked(true)]),
        widget(funnel, funnel_chart, [title("Sales Funnel")]),
        widget(leaderboard, ranked_list, [title("Top Performers"), limit(5)])
    ]),
    features([
        date_range_picker,
        team_filter,
        export_csv
    ])
]).

template(realtime_monitor, dashboard, [
    title("Real-time Monitor"),
    description("Live system metrics and alerts"),
    layout(grid),
    grid_config([
        columns(4),
        gap('0.5rem')
    ]),
    widgets([
        widget(cpu, gauge, [metric(cpu_usage), threshold([warning(70), critical(90)])]),
        widget(memory, gauge, [metric(memory_usage), threshold([warning(80), critical(95)])]),
        widget(network, gauge, [metric(network_io), unit('MB/s')]),
        widget(requests, counter, [metric(requests_per_sec), format('0,0')]),
        widget(latency, sparkline, [metric(response_time), window(60)]),
        widget(errors, sparkline, [metric(error_rate), window(60), alert(true)]),
        widget(logs, log_stream, [limit(100), autoscroll(true)]),
        widget(alerts, alert_list, [severity([critical, warning])])
    ]),
    features([
        auto_refresh(5000),
        alert_sounds,
        dark_mode
    ]),
    theme(dark)
]).

% ============================================================================
% REPORT TEMPLATES
% ============================================================================

template(monthly_report, report, [
    title("Monthly Report"),
    description("Executive summary with key metrics"),
    layout(vertical),
    sections([
        section(header, [
            title("Monthly Performance Report"),
            subtitle("{month} {year}"),
            logo(true)
        ]),
        section(summary, [
            title("Executive Summary"),
            content(text_block),
            metrics([revenue, growth, customers])
        ]),
        section(performance, [
            title("Performance Overview"),
            charts([
                chart(trend, line_chart, [data(monthly_trend)]),
                chart(breakdown, pie_chart, [data(category_breakdown)])
            ])
        ]),
        section(details, [
            title("Detailed Analysis"),
            table(performance_table, [columns([name, value, change, status])])
        ]),
        section(footer, [
            page_numbers(true),
            generated_date(true)
        ])
    ]),
    features([
        export_pdf,
        print_optimized
    ]),
    print_config([
        page_size(a4),
        orientation(portrait),
        margins([top(20), bottom(20), left(15), right(15)])
    ])
]).

template(comparison_report, report, [
    title("Comparison Report"),
    description("Side-by-side metric comparison"),
    layout(vertical),
    sections([
        section(header, [title("Comparison Analysis")]),
        section(comparison, [
            type(side_by_side),
            items(2),
            charts([
                chart(metric1, bar_chart, [grouped(true)]),
                chart(metric2, radar_chart, [])
            ])
        ]),
        section(table, [
            title("Detailed Comparison"),
            type(comparison_table)
        ])
    ]),
    features([export_pdf])
]).

% ============================================================================
% EXPLORER TEMPLATES
% ============================================================================

template(data_explorer, explorer, [
    title("Data Explorer"),
    description("Interactive data exploration and filtering"),
    layout(sidebar_content),
    sidebar([
        filters([
            filter(date_range, date_picker, [label("Date Range")]),
            filter(category, multi_select, [label("Category"), options(dynamic)]),
            filter(status, checkbox_group, [label("Status")]),
            filter(search, text_input, [label("Search"), debounce(300)])
        ]),
        saved_views(true),
        export_options([csv, json, excel])
    ]),
    content([
        tabs([
            tab(chart, [title("Visualization"), content(dynamic_chart)]),
            tab(table, [title("Data Table"), content(data_table)]),
            tab(raw, [title("Raw Data"), content(json_viewer)])
        ])
    ]),
    features([
        column_selector,
        sort_controls,
        pagination,
        row_selection,
        bulk_actions
    ])
]).

template(chart_explorer, explorer, [
    title("Chart Explorer"),
    description("Explore and customize chart configurations"),
    layout(split),
    left_panel([
        chart_type_selector,
        data_mapping([
            x_axis_selector,
            y_axis_selector,
            series_selector,
            color_selector
        ]),
        style_options([
            theme_selector,
            color_palette,
            legend_position,
            axis_options
        ])
    ]),
    right_panel([
        live_preview(true),
        code_export([react, vanilla_js, python])
    ]),
    features([
        save_configuration,
        share_link,
        embed_code
    ])
]).

% ============================================================================
% PRESENTATION TEMPLATES
% ============================================================================

template(slide_deck, presentation, [
    title("Presentation"),
    description("Slide-based data presentation"),
    layout(slides),
    slide_config([
        aspect_ratio('16:9'),
        transition(fade),
        duration(500)
    ]),
    slides([
        slide(title_slide, [
            title("{presentation_title}"),
            subtitle("{date}"),
            background(gradient)
        ]),
        slide(overview, [
            title("Overview"),
            content(bullet_points),
            animation(fade_in_up)
        ]),
        slide(chart_slide, [
            title("{chart_title}"),
            chart(dynamic),
            animation(chart_draw)
        ]),
        slide(comparison_slide, [
            title("Comparison"),
            layout(two_column),
            charts([left_chart, right_chart])
        ]),
        slide(conclusion, [
            title("Key Takeaways"),
            content(bullet_points),
            animation(fade_in_up)
        ])
    ]),
    features([
        presenter_mode,
        keyboard_navigation,
        progress_indicator,
        fullscreen
    ])
]).

% ============================================================================
% TEMPLATE JSX GENERATION
% ============================================================================

%% generate_template_jsx(+Name, -JSX)
%  Generate React JSX for a template.
generate_template_jsx(Name, JSX) :-
    template(Name, Type, Options),
    atom_string(Name, NameStr),
    generate_component_name(Name, ComponentName),
    (member(title(Title), Options) -> true ; Title = "Dashboard"),
    generate_template_body(Type, Options, Body),
    generate_template_imports(Type, Options, Imports),
    format(atom(JSX), '~w

interface ~wProps {
  data?: unknown;
  onRefresh?: () => void;
  className?: string;
}

export const ~w: React.FC<~wProps> = ({
  data,
  onRefresh,
  className = ""
}) => {
  return (
    <div className={`template-~w ${className}`}>
      <header className="template-header">
        <h1>~w</h1>
        {onRefresh && (
          <button onClick={onRefresh} className="refresh-btn">
            Refresh
          </button>
        )}
      </header>
      <main className="template-content">
~w
      </main>
    </div>
  );
};
', [Imports, ComponentName, ComponentName, ComponentName, NameStr, Title, Body]).

generate_component_name(Name, ComponentName) :-
    atom_string(Name, NameStr),
    split_string(NameStr, "_", "", Parts),
    maplist(capitalize_first, Parts, CapParts),
    atomic_list_concat(CapParts, '', ComponentName).

capitalize_first(Str, Cap) :-
    string_chars(Str, [First|Rest]),
    upcase_atom(First, Upper),
    atom_chars(Upper, [UpperChar]),
    atom_chars(RestAtom, Rest),
    format(atom(Cap), '~w~w', [UpperChar, RestAtom]).

generate_template_imports(dashboard, Options, Imports) :-
    (member(widgets(Widgets), Options) -> true ; Widgets = []),
    findall(Import, (
        member(widget(_, WidgetType, _), Widgets),
        widget_import(WidgetType, Import)
    ), ImportList),
    list_to_set(ImportList, UniqueImports),
    atomic_list_concat(['import React from "react";' | UniqueImports], '\n', Imports).

generate_template_imports(_, _, 'import React from "react";').

widget_import(kpi_card, 'import { KPICard } from "./widgets/KPICard";').
widget_import(line_chart, 'import { LineChart } from "./charts/LineChart";').
widget_import(bar_chart, 'import { BarChart } from "./charts/BarChart";').
widget_import(pie_chart, 'import { PieChart } from "./charts/PieChart";').
widget_import(data_table, 'import { DataTable } from "./widgets/DataTable";').
widget_import(gauge, 'import { Gauge } from "./widgets/Gauge";').
widget_import(sparkline, 'import { Sparkline } from "./widgets/Sparkline";').
widget_import(_, '').

generate_template_body(dashboard, Options, Body) :-
    (member(widgets(Widgets), Options) -> true ; Widgets = []),
    (member(grid_config(GridConfig), Options) -> true ; GridConfig = []),
    findall(WidgetJSX, (
        member(widget(Id, Type, WidgetOpts), Widgets),
        generate_widget_jsx(Id, Type, WidgetOpts, WidgetJSX)
    ), WidgetJSXList),
    atomic_list_concat(WidgetJSXList, '\n        ', WidgetsStr),
    format(atom(Body), '        <div className="dashboard-grid">
        ~w
        </div>', [WidgetsStr]).

generate_template_body(report, Options, Body) :-
    (member(sections(Sections), Options) -> true ; Sections = []),
    findall(SectionJSX, (
        member(section(Id, SectionOpts), Sections),
        generate_section_jsx(Id, SectionOpts, SectionJSX)
    ), SectionJSXList),
    atomic_list_concat(SectionJSXList, '\n        ', SectionsStr),
    format(atom(Body), '        <article className="report">
        ~w
        </article>', [SectionsStr]).

generate_template_body(explorer, Options, Body) :-
    format(atom(Body), '        <div className="explorer-layout">
          <aside className="explorer-sidebar">
            <div className="filters">
              {/* Filter components */}
            </div>
          </aside>
          <section className="explorer-content">
            {/* Content area */}
          </section>
        </div>', []).

generate_template_body(_, _, '        {/* Template content */}').

generate_widget_jsx(Id, Type, Opts, JSX) :-
    atom_string(Id, IdStr),
    atom_string(Type, TypeStr),
    (member(title(Title), Opts) -> true ; Title = ""),
    format(atom(JSX), '<div className="widget widget-~w" data-widget="~w">
            <~w title="~w" />
          </div>', [IdStr, TypeStr, TypeStr, Title]).

generate_section_jsx(Id, Opts, JSX) :-
    atom_string(Id, IdStr),
    (member(title(Title), Opts) -> true ; Title = ""),
    format(atom(JSX), '<section className="report-section section-~w">
            <h2>~w</h2>
            {/* Section content */}
          </section>', [IdStr, Title]).

% ============================================================================
% TEMPLATE CSS GENERATION
% ============================================================================

%% generate_template_css(+Name, -CSS)
%  Generate CSS for a template.
generate_template_css(Name, CSS) :-
    template(Name, Type, Options),
    atom_string(Name, NameStr),
    generate_base_css(Type, BaseCSS),
    generate_layout_css(Options, LayoutCSS),
    generate_responsive_css(Options, ResponsiveCSS),
    format(atom(CSS), '/* Template: ~w */

.template-~w {
  width: 100%;
  min-height: 100vh;
  background: var(--color-background, #f8fafc);
}

.template-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md, 16px) var(--spacing-lg, 24px);
  background: var(--color-surface, #ffffff);
  border-bottom: 1px solid var(--color-border, #e2e8f0);
}

.template-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--color-text-primary, #0f172a);
}

.refresh-btn {
  padding: var(--spacing-xs, 4px) var(--spacing-sm, 8px);
  border: 1px solid var(--color-border, #e2e8f0);
  border-radius: var(--border-radius-sm, 4px);
  background: var(--color-surface, #ffffff);
  cursor: pointer;
  transition: background 150ms ease;
}

.refresh-btn:hover {
  background: var(--color-background, #f8fafc);
}

.template-content {
  padding: var(--spacing-lg, 24px);
}

~w

~w

~w
', [NameStr, NameStr, BaseCSS, LayoutCSS, ResponsiveCSS]).

generate_base_css(dashboard, CSS) :-
    format(atom(CSS), '/* Dashboard Base Styles */
.dashboard-grid {
  display: grid;
  gap: var(--spacing-md, 16px);
}

.widget {
  background: var(--color-surface, #ffffff);
  border-radius: var(--border-radius-md, 8px);
  padding: var(--spacing-md, 16px);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.05));
}

.widget-kpi_card {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.widget-line_chart,
.widget-bar_chart,
.widget-pie_chart {
  min-height: 300px;
}

.widget-data_table {
  overflow: auto;
}', []).

generate_base_css(report, CSS) :-
    format(atom(CSS), '/* Report Base Styles */
.report {
  max-width: 800px;
  margin: 0 auto;
  background: var(--color-surface, #ffffff);
  padding: var(--spacing-xl, 32px);
}

.report-section {
  margin-bottom: var(--spacing-xl, 32px);
  page-break-inside: avoid;
}

.report-section h2 {
  margin-bottom: var(--spacing-md, 16px);
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--color-text-primary, #0f172a);
  border-bottom: 2px solid var(--color-primary, #3b82f6);
  padding-bottom: var(--spacing-xs, 4px);
}', []).

generate_base_css(explorer, CSS) :-
    format(atom(CSS), '/* Explorer Base Styles */
.explorer-layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: calc(100vh - 60px);
}

.explorer-sidebar {
  background: var(--color-surface, #ffffff);
  border-right: 1px solid var(--color-border, #e2e8f0);
  padding: var(--spacing-md, 16px);
  overflow-y: auto;
}

.explorer-content {
  padding: var(--spacing-md, 16px);
  overflow: auto;
}

.filters {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md, 16px);
}', []).

generate_base_css(_, '').

generate_layout_css(Options, CSS) :-
    (member(grid_config(GridConfig), Options) ->
        (member(columns(Cols), GridConfig) -> true ; Cols = 4),
        format(atom(CSS), '.dashboard-grid {
  grid-template-columns: repeat(~w, 1fr);
}', [Cols])
    ;
        CSS = ''
    ).

generate_responsive_css(Options, CSS) :-
    (member(responsive(Breakpoints), Options) ->
        findall(BpCSS, (
            member(breakpoint(Size, BpOpts), Breakpoints),
            generate_breakpoint_css(Size, BpOpts, BpCSS)
        ), BpCSSList),
        atomic_list_concat(BpCSSList, '\n\n', CSS)
    ;
        CSS = '/* Responsive Styles */
@media (max-width: 768px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
  .explorer-layout {
    grid-template-columns: 1fr;
  }
  .explorer-sidebar {
    border-right: none;
    border-bottom: 1px solid var(--color-border, #e2e8f0);
  }
}'
    ).

generate_breakpoint_css(mobile, Opts, CSS) :-
    (member(columns(Cols), Opts) -> true ; Cols = 1),
    format(atom(CSS), '@media (max-width: 480px) {
  .dashboard-grid { grid-template-columns: repeat(~w, 1fr); }
}', [Cols]).
generate_breakpoint_css(tablet, Opts, CSS) :-
    (member(columns(Cols), Opts) -> true ; Cols = 2),
    format(atom(CSS), '@media (min-width: 481px) and (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: repeat(~w, 1fr); }
}', [Cols]).
generate_breakpoint_css(desktop, Opts, CSS) :-
    (member(columns(Cols), Opts) -> true ; Cols = 4),
    format(atom(CSS), '@media (min-width: 1025px) {
  .dashboard-grid { grid-template-columns: repeat(~w, 1fr); }
}', [Cols]).

%% generate_print_styles(-CSS)
%  Generate print-optimized CSS.
generate_print_styles(CSS) :-
    format(atom(CSS), '@media print {
  .template-header .refresh-btn {
    display: none;
  }

  .report {
    max-width: none;
    padding: 0;
    box-shadow: none;
  }

  .report-section {
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .widget {
    box-shadow: none;
    border: 1px solid #e2e8f0;
  }

  @page {
    margin: 2cm;
  }

  @page :first {
    margin-top: 0;
  }
}', []).

% ============================================================================
% TEMPLATE TYPES GENERATION
% ============================================================================

%% generate_template_types(+Name, -Types)
%  Generate TypeScript types for a template.
generate_template_types(Name, Types) :-
    template(Name, Type, Options),
    generate_component_name(Name, ComponentName),
    (member(widgets(Widgets), Options) ->
        generate_widget_types(Widgets, WidgetTypes)
    ;
        WidgetTypes = ''
    ),
    format(atom(Types), 'export interface ~wProps {
  data?: ~wData;
  onRefresh?: () => void;
  onWidgetClick?: (widgetId: string) => void;
  className?: string;
  theme?: "light" | "dark";
}

export interface ~wData {
  // Define your data structure
  [key: string]: unknown;
}

~w
', [ComponentName, ComponentName, ComponentName, WidgetTypes]).

generate_widget_types(Widgets, Types) :-
    findall(WType, (
        member(widget(Id, Type, _), Widgets),
        atom_string(Id, IdStr),
        format(atom(WType), 'export interface Widget~wData {\n  // Widget-specific data\n}', [IdStr])
    ), TypeList),
    atomic_list_concat(TypeList, '\n\n', Types).

% ============================================================================
% TEMPLATE HOOK GENERATION
% ============================================================================

%% generate_template_hook(+Name, -Hook)
%  Generate a React hook for template data management.
generate_template_hook(Name, Hook) :-
    template(Name, _Type, Options),
    generate_component_name(Name, ComponentName),
    (member(features(Features), Options) -> true ; Features = []),
    (member(auto_refresh(Interval), Features) -> RefreshCode = generate_refresh_code(Interval) ; RefreshCode = ''),
    format(atom(Hook), 'import { useState, useEffect, useCallback } from "react";

interface Use~wResult {
  data: unknown;
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  lastUpdated: Date | null;
}

export const use~w = (dataSource?: string): Use~wResult => {
  const [data, setData] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(dataSource || "/api/dashboard/data");
      if (!response.ok) throw new Error("Failed to fetch data");
      const result = await response.json();
      setData(result);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, [dataSource]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { data, loading, error, refresh, lastUpdated };
};
', [ComponentName, ComponentName, ComponentName]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_template(+Name, -Spec)
%  Get template specification.
get_template(Name, spec(Name, Type, Options)) :-
    template(Name, Type, Options).

%% list_templates(-Templates)
%  List all templates.
list_templates(Templates) :-
    findall(Name, template(Name, _, _), Templates).

%% list_templates_by_type(+Type, -Templates)
%  List templates of a specific type.
list_templates_by_type(Type, Templates) :-
    findall(Name, template(Name, Type, _), Templates).

%% template_has_feature(+Name, +Feature)
%  Check if template has a feature.
template_has_feature(Name, Feature) :-
    template(Name, _, Options),
    member(features(Features), Options),
    member(Feature, Features).

%% get_template_charts(+Name, -Charts)
%  Get list of charts in a template.
get_template_charts(Name, Charts) :-
    template(Name, _, Options),
    member(widgets(Widgets), Options),
    findall(Type, (
        member(widget(_, Type, _), Widgets),
        chart_type(Type)
    ), Charts).

chart_type(line_chart).
chart_type(bar_chart).
chart_type(pie_chart).
chart_type(area_chart).
chart_type(scatter_chart).
chart_type(radar_chart).
chart_type(funnel_chart).

%% generate_dashboard_layout(+Spec, -Layout)
%  Generate dashboard grid layout CSS.
generate_dashboard_layout(Spec, Layout) :-
    member(grid_config(Config), Spec),
    (member(areas(Areas), Config) ->
        generate_grid_areas(Areas, AreasCSS),
        format(atom(Layout), 'grid-template-areas: ~w;', [AreasCSS])
    ;
        Layout = ''
    ).

generate_grid_areas(Areas, CSS) :-
    findall(Row, (
        member(RowAreas, Areas),
        atomic_list_concat(RowAreas, ' ', RowStr),
        format(atom(Row), '"~w"', [RowStr])
    ), Rows),
    atomic_list_concat(Rows, ' ', CSS).

%% generate_dashboard_widgets(+Spec, -Widgets)
%  Generate widget specifications.
generate_dashboard_widgets(Spec, Widgets) :-
    member(widgets(Widgets), Spec), !.
generate_dashboard_widgets(_, []).

%% generate_report_layout(+Spec, -Layout)
%  Generate report layout.
generate_report_layout(Spec, Layout) :-
    member(print_config(PrintConfig), Spec),
    (member(page_size(Size), PrintConfig) -> true ; Size = a4),
    (member(orientation(Orient), PrintConfig) -> true ; Orient = portrait),
    format(atom(Layout), '@page { size: ~w ~w; }', [Size, Orient]).

% ============================================================================
% TEMPLATE GENERATION
% ============================================================================

%% generate_template(+Name, -Code)
%  Generate complete template code bundle.
generate_template(Name, Code) :-
    generate_template_jsx(Name, JSX),
    generate_template_css(Name, CSS),
    generate_template_types(Name, Types),
    generate_template_hook(Name, Hook),
    format(atom(Code), '// =====================================
// Template: ~w
// =====================================

// --- Types ---
~w

// --- Hook ---
~w

// --- Component ---
~w

// --- Styles ---
/*
~w
*/
', [Name, Types, Hook, JSX, CSS]).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_template(+Name, +Type, +Options)
%  Declare a custom template.
declare_template(Name, Type, Options) :-
    retractall(template(Name, _, _)),
    assertz(template(Name, Type, Options)).

%% clear_templates/0
%  Clear custom templates.
clear_templates :-
    retractall(template(_, _, _)).

% ============================================================================
% TESTING
% ============================================================================

test_template_library :-
    writeln('Testing template library...'),

    % Test template existence
    (template(analytics_dashboard, dashboard, _) ->
        writeln('  [PASS] analytics_dashboard exists') ;
        writeln('  [FAIL] analytics_dashboard')),

    % Test template types
    (template_type(dashboard, _) ->
        writeln('  [PASS] dashboard type defined') ;
        writeln('  [FAIL] dashboard type')),

    % Test listing
    (list_templates(Templates), length(Templates, TL), TL > 3 ->
        writeln('  [PASS] has multiple templates') ;
        writeln('  [FAIL] template count')),

    % Test JSX generation
    (generate_template_jsx(analytics_dashboard, JSX), atom_length(JSX, JL), JL > 500 ->
        writeln('  [PASS] generate_template_jsx works') ;
        writeln('  [FAIL] generate_template_jsx')),

    % Test CSS generation
    (generate_template_css(analytics_dashboard, CSS), atom_length(CSS, CL), CL > 500 ->
        writeln('  [PASS] generate_template_css works') ;
        writeln('  [FAIL] generate_template_css')),

    % Test feature check
    (template_has_feature(analytics_dashboard, date_range_picker) ->
        writeln('  [PASS] has date_range_picker feature') ;
        writeln('  [FAIL] feature check')),

    writeln('Template library tests complete.').

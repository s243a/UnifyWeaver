% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Spec Validation - Declarative Specification Validation with Error Reporting
%
% This module provides comprehensive validation for visualization and layout
% specifications with detailed error messages and suggestions.
%
% Usage:
%   % Validate a curve specification
%   ?- validate_curve(my_curve, Errors).
%   Errors = []  % No errors - valid
%
%   % Validate with detailed report
%   ?- validation_report(my_curve, curve, Report).
%
%   % Validate entire dashboard
%   ?- validate_dashboard(sales_dashboard, Errors).

:- module(spec_validation, [
    % Core validation predicates
    validate_curve/2,                   % validate_curve(+Name, -Errors)
    validate_plot_spec/2,               % validate_plot_spec(+Name, -Errors)
    validate_layout/2,                  % validate_layout(+Name, -Errors)
    validate_theme/2,                   % validate_theme(+Name, -Errors)
    validate_gauge/2,                   % validate_gauge(+Name, -Errors)
    validate_funnel/2,                  % validate_funnel(+Name, -Errors)
    validate_dashboard/2,               % validate_dashboard(+Name, -Errors)

    % Batch validation
    validate_all_specs/1,               % validate_all_specs(-AllErrors)
    validate_specs/2,                   % validate_specs(+SpecNames, -Errors)

    % Reporting
    validation_report/3,                % validation_report(+Name, +Type, -Report)
    format_errors/2,                    % format_errors(+Errors, -FormattedString)
    format_error/2,                     % format_error(+Error, -String)

    % Validation utilities
    valid_color/1,                      % valid_color(+Color)
    valid_css_unit/1,                   % valid_css_unit(+Value)
    valid_number_range/3,               % valid_number_range(+Value, +Min, +Max)
    valid_identifier/1,                 % valid_identifier(+Name)

    % Safe generation (validate before generate)
    safe_generate/4,                    % safe_generate(+Type, +Name, +Generator, -Result)

    % Testing
    test_spec_validation/0
]).

:- use_module(library(lists)).

% ============================================================================
% ERROR STRUCTURE
% ============================================================================

% Error structure: error(Severity, Code, Message, Context)
%   Severity: error | warning | info
%   Code: Atom like 'missing_property', 'invalid_value', etc.
%   Message: Human-readable error message
%   Context: List of key-value pairs for additional context

% ============================================================================
% VALIDATION: CURVES
% ============================================================================

%% validate_curve(+Name, -Errors) is det.
%
% Validates a curve specification.
%
validate_curve(Name, Errors) :-
    (curve_spec_exists(Name, Props)
    ->  validate_curve_properties(Name, Props, Errors)
    ;   Errors = [error(error, not_found, "Curve specification not found", [name(Name)])]
    ).

curve_spec_exists(Name, Props) :-
    catch(
        (curve_plot_generator:curve(Name, Props) ; curve(Name, Props)),
        _,
        fail
    ).

validate_curve_properties(Name, Props, Errors) :-
    findall(Error, validate_curve_property(Name, Props, Error), Errors).

validate_curve_property(_, Props, error(error, missing_type, "Curve must have a type", [])) :-
    \+ member(type(_), Props).

validate_curve_property(_, Props, error(error, invalid_type, "Invalid curve type", [type(Type), valid_types(ValidTypes)])) :-
    member(type(Type), Props),
    valid_curve_types(ValidTypes),
    \+ member(Type, ValidTypes).

validate_curve_property(_, Props, error(warning, missing_color, "Curve has no color specified, will use default", [])) :-
    \+ member(color(_), Props).

validate_curve_property(_, Props, error(error, invalid_color, "Invalid color format", [color(Color)])) :-
    member(color(Color), Props),
    \+ valid_color(Color).

validate_curve_property(_, Props, error(error, invalid_amplitude, "Amplitude must be positive", [amplitude(A)])) :-
    member(amplitude(A), Props),
    A =< 0.

validate_curve_property(_, Props, error(warning, no_label, "Curve has no label for legend", [])) :-
    \+ member(label(_), Props).

valid_curve_types([linear, quadratic, cubic, sine, cosine, exponential, absolute, custom]).

% ============================================================================
% VALIDATION: PLOT SPECIFICATIONS
% ============================================================================

%% validate_plot_spec(+Name, -Errors) is det.
%
% Validates a plot specification.
%
validate_plot_spec(Name, Errors) :-
    (plot_spec_exists(Name, Config)
    ->  validate_plot_config(Name, Config, Errors)
    ;   Errors = [error(error, not_found, "Plot specification not found", [name(Name)])]
    ).

plot_spec_exists(Name, Config) :-
    catch(
        (curve_plot_generator:plot_spec(Name, Config) ; plot_spec(Name, Config)),
        _,
        fail
    ).

validate_plot_config(Name, Config, Errors) :-
    findall(Error, validate_plot_property(Name, Config, Error), Errors).

validate_plot_property(_, Config, error(warning, no_title, "Plot has no title", [])) :-
    \+ member(title(_), Config).

validate_plot_property(_, Config, error(error, no_curves, "Plot must reference at least one curve", [])) :-
    \+ member(curves(_), Config).

validate_plot_property(_, Config, error(error, empty_curves, "Curves list is empty", [])) :-
    member(curves([]), Config).

validate_plot_property(Name, Config, error(error, missing_curve, "Referenced curve does not exist", [plot(Name), curve(CurveName)])) :-
    member(curves(Curves), Config),
    member(CurveName, Curves),
    \+ curve_spec_exists(CurveName, _).

validate_plot_property(_, Config, error(error, invalid_x_range, "x_range must be (Min, Max) where Min < Max", [x_range(Min, Max)])) :-
    member(x_range(Min, Max), Config),
    Min >= Max.

validate_plot_property(_, Config, error(error, invalid_y_range, "y_range must be (Min, Max) where Min < Max", [y_range(Min, Max)])) :-
    member(y_range(Min, Max), Config),
    Min >= Max.

validate_plot_property(_, Config, error(error, invalid_theme, "Unknown theme", [theme(Theme), valid_themes(ValidThemes)])) :-
    member(theme(Theme), Config),
    valid_themes(ValidThemes),
    \+ member(Theme, ValidThemes).

valid_themes([light, dark, high_contrast, brand]).

% ============================================================================
% VALIDATION: LAYOUTS
% ============================================================================

%% validate_layout(+Name, -Errors) is det.
%
% Validates a layout specification.
%
validate_layout(Name, Errors) :-
    (layout_spec_exists(Name, Strategy, Options)
    ->  validate_layout_config(Name, Strategy, Options, Errors)
    ;   Errors = [error(error, not_found, "Layout specification not found", [name(Name)])]
    ).

layout_spec_exists(Name, Strategy, Options) :-
    catch(
        (layout_generator:layout(Name, Strategy, Options) ; layout(Name, Strategy, Options)),
        _,
        fail
    ).

validate_layout_config(Name, Strategy, Options, Errors) :-
    findall(Error, validate_layout_property(Name, Strategy, Options, Error), Errors).

validate_layout_property(_, Strategy, _, error(error, invalid_strategy, "Unknown layout strategy", [strategy(Strategy), valid_strategies(ValidStrategies)])) :-
    valid_layout_strategies(ValidStrategies),
    \+ member(Strategy, ValidStrategies).

validate_layout_property(_, grid, Options, error(error, missing_grid_def, "Grid layout requires 'areas' or 'columns'", [])) :-
    \+ member(areas(_), Options),
    \+ member(columns(_), Options).

validate_layout_property(_, grid, Options, error(error, inconsistent_areas, "Grid areas must have consistent column counts", [row_lengths(Lengths)])) :-
    member(areas(Rows), Options),
    findall(Len, (member(Row, Rows), length(Row, Len)), Lengths),
    \+ all_equal(Lengths).

validate_layout_property(_, grid, Options, error(warning, missing_gap, "Grid has no gap specified", [])) :-
    \+ member(gap(_), Options).

validate_layout_property(_, flex, Options, error(warning, no_direction, "Flex layout has no direction, defaults to row", [])) :-
    \+ member(direction(_), Options).

validate_layout_property(_, _, Options, error(error, invalid_gap, "Gap must be a valid CSS value", [gap(Gap)])) :-
    member(gap(Gap), Options),
    \+ valid_css_unit(Gap).

valid_layout_strategies([grid, flex, stack, responsive]).

all_equal([]).
all_equal([_]).
all_equal([X, X | Rest]) :- all_equal([X | Rest]).

% ============================================================================
% VALIDATION: THEMES
% ============================================================================

%% validate_theme(+Name, -Errors) is det.
%
% Validates a theme specification.
%
validate_theme(Name, Errors) :-
    (theme_spec_exists(Name, Config)
    ->  validate_theme_config(Name, Config, Errors)
    ;   Errors = [error(error, not_found, "Theme specification not found", [name(Name)])]
    ).

theme_spec_exists(Name, Config) :-
    catch(
        (theme_generator:theme(Name, Config) ; theme(Name, Config)),
        _,
        fail
    ).

validate_theme_config(Name, Config, Errors) :-
    findall(Error, validate_theme_property(Name, Config, Error), Errors).

validate_theme_property(_, Config, error(warning, no_colors, "Theme has no color definitions", [])) :-
    \+ member(colors(_), Config).

validate_theme_property(_, Config, error(error, invalid_color_value, "Invalid color value in theme", [key(Key), value(Value)])) :-
    member(colors(Colors), Config),
    member(color(Key, Value), Colors),
    \+ valid_color(Value).

validate_theme_property(_, Config, error(warning, missing_accessibility, "Theme has no accessibility settings", [])) :-
    \+ member(accessibility(_), Config).

validate_theme_property(_, Config, error(error, invalid_contrast, "Contrast ratio must be >= 4.5 for AA compliance", [ratio(Ratio)])) :-
    member(accessibility(Access), Config),
    member(min_contrast_ratio(Ratio), Access),
    Ratio < 4.5.

% ============================================================================
% VALIDATION: GAUGES
% ============================================================================

%% validate_gauge(+Name, -Errors) is det.
%
% Validates a gauge specification.
%
validate_gauge(Name, Errors) :-
    (gauge_spec_exists(Name, Config)
    ->  validate_gauge_config(Name, Config, Errors)
    ;   Errors = [error(error, not_found, "Gauge specification not found", [name(Name)])]
    ).

gauge_spec_exists(Name, Config) :-
    catch(
        (gauge_chart_generator:gauge_spec(Name, Config) ; gauge_spec(Name, Config)),
        _,
        fail
    ).

validate_gauge_config(Name, Config, Errors) :-
    findall(Error, validate_gauge_property(Name, Config, Error), Errors).

validate_gauge_property(_, Config, error(error, missing_value, "Gauge must have a value", [])) :-
    \+ member(value(_), Config).

validate_gauge_property(_, Config, error(error, missing_range, "Gauge needs min and max values", [])) :-
    (\+ member(min(_), Config) ; \+ member(max(_), Config)).

validate_gauge_property(_, Config, error(error, invalid_range, "min must be less than max", [min(Min), max(Max)])) :-
    member(min(Min), Config),
    member(max(Max), Config),
    Min >= Max.

validate_gauge_property(_, Config, error(error, value_out_of_range, "Value is outside min/max range", [value(V), min(Min), max(Max)])) :-
    member(value(V), Config),
    member(min(Min), Config),
    member(max(Max), Config),
    (V < Min ; V > Max).

validate_gauge_property(_, Config, error(warning, no_thresholds, "Gauge has no threshold colors defined", [])) :-
    \+ member(thresholds(_), Config).

validate_gauge_property(_, Config, error(error, overlapping_thresholds, "Threshold ranges overlap", [threshold1(T1), threshold2(T2)])) :-
    member(thresholds(Thresholds), Config),
    member(threshold(_, S1, E1, _), Thresholds),
    member(threshold(_, S2, E2, _), Thresholds),
    S1 \= S2,  % Different thresholds
    ranges_overlap(S1, E1, S2, E2),
    T1 = range(S1, E1),
    T2 = range(S2, E2).

ranges_overlap(S1, E1, S2, E2) :-
    S1 < E2, S2 < E1.

% ============================================================================
% VALIDATION: FUNNELS
% ============================================================================

%% validate_funnel(+Name, -Errors) is det.
%
% Validates a funnel specification.
%
validate_funnel(Name, Errors) :-
    (funnel_spec_exists(Name, Config)
    ->  validate_funnel_config(Name, Config, Errors)
    ;   Errors = [error(error, not_found, "Funnel specification not found", [name(Name)])]
    ).

funnel_spec_exists(Name, Config) :-
    catch(
        (funnel_chart_generator:funnel_spec(Name, Config) ; funnel_spec(Name, Config)),
        _,
        fail
    ).

validate_funnel_config(Name, Config, Errors) :-
    findall(Error, validate_funnel_property(Name, Config, Error), Errors).

validate_funnel_property(_, Config, error(warning, no_title, "Funnel has no title", [])) :-
    \+ member(title(_), Config).

validate_funnel_property(Name, _, error(error, no_stages, "Funnel has no stages defined", [funnel(Name)])) :-
    \+ funnel_has_stages(Name).

validate_funnel_property(Name, _, error(warning, single_stage, "Funnel has only one stage", [funnel(Name)])) :-
    funnel_stage_count(Name, 1).

validate_funnel_property(Name, _, error(error, non_decreasing_values, "Funnel values should decrease (top to bottom)", [funnel(Name)])) :-
    \+ funnel_values_decreasing(Name).

funnel_has_stages(Name) :-
    catch(
        funnel_chart_generator:funnel_stage(Name, _, _),
        _,
        fail
    ).

funnel_stage_count(Name, Count) :-
    findall(Stage, funnel_chart_generator:funnel_stage(Name, Stage, _), Stages),
    length(Stages, Count).

funnel_values_decreasing(Name) :-
    findall(V, (
        funnel_chart_generator:funnel_stage(Name, _, Props),
        member(value(V), Props)
    ), Values),
    (Values = [] ; is_decreasing(Values)).

is_decreasing([]).
is_decreasing([_]).
is_decreasing([A, B | Rest]) :-
    A >= B,
    is_decreasing([B | Rest]).

% ============================================================================
% VALIDATION: DASHBOARDS
% ============================================================================

%% validate_dashboard(+Name, -Errors) is det.
%
% Validates a complete dashboard specification including all referenced components.
%
validate_dashboard(Name, Errors) :-
    findall(Error, validate_dashboard_component(Name, Error), ComponentErrors),
    validate_dashboard_layout(Name, LayoutErrors),
    append(ComponentErrors, LayoutErrors, Errors).

validate_dashboard_component(Name, error(error, missing_layout, "Dashboard has no layout defined", [dashboard(Name)])) :-
    \+ dashboard_has_layout(Name).

validate_dashboard_component(Name, Error) :-
    dashboard_uses_visualization(Name, VizName, VizType),
    validate_visualization(VizType, VizName, VizErrors),
    VizErrors \= [],
    member(Error, VizErrors).

validate_dashboard_layout(Name, Errors) :-
    (dashboard_has_layout(Name)
    ->  dashboard_layout(Name, LayoutName),
        validate_layout(LayoutName, Errors)
    ;   Errors = []
    ).

dashboard_has_layout(_) :- fail.  % Placeholder - override in actual implementation
dashboard_layout(_, _) :- fail.   % Placeholder
dashboard_uses_visualization(_, _, _) :- fail.  % Placeholder

validate_visualization(curve, Name, Errors) :- validate_curve(Name, Errors).
validate_visualization(gauge, Name, Errors) :- validate_gauge(Name, Errors).
validate_visualization(funnel, Name, Errors) :- validate_funnel(Name, Errors).
validate_visualization(_, _, []).

% ============================================================================
% BATCH VALIDATION
% ============================================================================

%% validate_all_specs(-AllErrors) is det.
%
% Validates all defined specifications in the system.
%
validate_all_specs(AllErrors) :-
    findall(curve_error(Name, Errors), (
        curve_spec_exists(Name, _),
        validate_curve(Name, Errors),
        Errors \= []
    ), CurveErrors),
    findall(layout_error(Name, Errors), (
        layout_spec_exists(Name, _, _),
        validate_layout(Name, Errors),
        Errors \= []
    ), LayoutErrors),
    append(CurveErrors, LayoutErrors, AllErrors).

%% validate_specs(+SpecNames, -Errors) is det.
%
% Validates a list of specifications by name.
%
validate_specs([], []).
validate_specs([Name-Type | Rest], AllErrors) :-
    validate_spec_by_type(Type, Name, Errors),
    validate_specs(Rest, RestErrors),
    append(Errors, RestErrors, AllErrors).

validate_spec_by_type(curve, Name, Errors) :- validate_curve(Name, Errors).
validate_spec_by_type(layout, Name, Errors) :- validate_layout(Name, Errors).
validate_spec_by_type(theme, Name, Errors) :- validate_theme(Name, Errors).
validate_spec_by_type(gauge, Name, Errors) :- validate_gauge(Name, Errors).
validate_spec_by_type(funnel, Name, Errors) :- validate_funnel(Name, Errors).
validate_spec_by_type(_, _, []).

% ============================================================================
% ERROR FORMATTING
% ============================================================================

%% validation_report(+Name, +Type, -Report) is det.
%
% Generates a human-readable validation report.
%
validation_report(Name, Type, Report) :-
    validate_spec_by_type(Type, Name, Errors),
    (Errors = []
    ->  format(atom(Report), "✓ ~w '~w' is valid", [Type, Name])
    ;   length(Errors, ErrorCount),
        count_by_severity(Errors, ErrorsByType),
        format_errors(Errors, FormattedErrors),
        format(atom(Report), "✗ ~w '~w' has ~w issue(s):~n~w~n~nSummary: ~w",
               [Type, Name, ErrorCount, FormattedErrors, ErrorsByType])
    ).

count_by_severity(Errors, Summary) :-
    include(is_error_severity, Errors, ErrorList),
    include(is_warning_severity, Errors, WarningList),
    length(ErrorList, ErrorCount),
    length(WarningList, WarningCount),
    format(atom(Summary), "~w error(s), ~w warning(s)", [ErrorCount, WarningCount]).

is_error_severity(error(error, _, _, _)).
is_warning_severity(error(warning, _, _, _)).

%% format_errors(+Errors, -FormattedString) is det.
%
% Formats a list of errors into a human-readable string.
%
format_errors(Errors, Formatted) :-
    maplist(format_error, Errors, Lines),
    atomic_list_concat(Lines, '\n', Formatted).

%% format_error(+Error, -String) is det.
%
% Formats a single error into a string.
%
format_error(error(Severity, Code, Message, Context), String) :-
    severity_symbol(Severity, Symbol),
    (Context = []
    ->  format(atom(String), "  ~w [~w] ~w", [Symbol, Code, Message])
    ;   format_context(Context, ContextStr),
        format(atom(String), "  ~w [~w] ~w (~w)", [Symbol, Code, Message, ContextStr])
    ).

severity_symbol(error, '✗').
severity_symbol(warning, '⚠').
severity_symbol(info, 'ℹ').

format_context(Context, String) :-
    maplist(format_context_item, Context, Items),
    atomic_list_concat(Items, ', ', String).

format_context_item(Item, String) :-
    (Item = (Key=Value)
    ->  format(atom(String), "~w=~w", [Key, Value])
    ;   Item =.. [Key, Value]
    ->  format(atom(String), "~w=~w", [Key, Value])
    ;   format(atom(String), "~w", [Item])
    ).

% ============================================================================
% VALIDATION UTILITIES
% ============================================================================

%% valid_color(+Color) is semidet.
%
% Checks if a color value is valid (hex, rgb, rgba, named).
%
valid_color(Color) :-
    atom(Color),
    (valid_hex_color(Color)
    ; valid_rgb_color(Color)
    ; valid_named_color(Color)
    ).

valid_hex_color(Color) :-
    atom_codes(Color, [0'# | Rest]),
    length(Rest, Len),
    member(Len, [3, 6, 8]),
    forall(member(C, Rest), is_hex_char(C)).

is_hex_char(C) :-
    (C >= 0'0, C =< 0'9)
    ; (C >= 0'a, C =< 0'f)
    ; (C >= 0'A, C =< 0'F).

valid_rgb_color(Color) :-
    atom(Color),
    (sub_atom(Color, 0, 4, _, 'rgb(')
    ; sub_atom(Color, 0, 5, _, 'rgba(')
    ).

valid_named_color(Color) :-
    member(Color, [
        red, green, blue, yellow, orange, purple, pink, cyan, magenta,
        black, white, gray, grey, transparent, inherit, currentColor
    ]).

%% valid_css_unit(+Value) is semidet.
%
% Checks if a value is a valid CSS unit value.
%
valid_css_unit(Value) :-
    atom(Value),
    atom_codes(Value, Codes),
    valid_css_unit_codes(Codes).

valid_css_unit_codes(Codes) :-
    % Number followed by unit
    append(NumCodes, UnitCodes, Codes),
    NumCodes \= [],
    forall(member(C, NumCodes), (is_digit(C) ; C =:= 0'. ; C =:= 0'-)),
    (UnitCodes = []  % unitless number
    ; valid_css_unit_suffix(UnitCodes)
    ).

is_digit(C) :- C >= 0'0, C =< 0'9.

valid_css_unit_suffix(Codes) :-
    atom_codes(Unit, Codes),
    member(Unit, [px, em, rem, '%', vh, vw, vmin, vmax, pt, cm, mm, in, fr, auto]).

%% valid_number_range(+Value, +Min, +Max) is semidet.
%
% Checks if a number is within a range.
%
valid_number_range(Value, Min, Max) :-
    number(Value),
    Value >= Min,
    Value =< Max.

%% valid_identifier(+Name) is semidet.
%
% Checks if a name is a valid identifier.
%
valid_identifier(Name) :-
    atom(Name),
    atom_codes(Name, [First | Rest]),
    (is_alpha(First) ; First =:= 0'_),
    forall(member(C, Rest), (is_alnum(C) ; C =:= 0'_ ; C =:= 0'-)).

is_alpha(C) :- (C >= 0'a, C =< 0'z) ; (C >= 0'A, C =< 0'Z).
is_alnum(C) :- is_alpha(C) ; is_digit(C).

% ============================================================================
% SAFE GENERATION
% ============================================================================

%% safe_generate(+Type, +Name, +Generator, -Result) is det.
%
% Validates a spec before calling the generator. Returns either:
%   success(Code) or failure(Errors)
%
safe_generate(Type, Name, Generator, Result) :-
    validate_spec_by_type(Type, Name, Errors),
    (Errors = []
    ->  (call(Generator, Name, Code)
        ->  Result = success(Code)
        ;   Result = failure([error(error, generation_failed, "Generator failed", [type(Type), name(Name)])])
        )
    ;   Result = failure(Errors)
    ).

% ============================================================================
% DYNAMIC PREDICATES (for testing)
% ============================================================================

:- dynamic curve/2.
:- dynamic plot_spec/2.
:- dynamic layout/3.
:- dynamic theme/2.
:- dynamic gauge_spec/2.
:- dynamic funnel_spec/2.

% ============================================================================
% TESTING
% ============================================================================

test_spec_validation :-
    format("Testing spec validation module...~n"),

    % Test valid color validation
    (valid_color('#ff0000') -> format("  ✓ valid_color hex~n") ; format("  ✗ valid_color hex~n")),
    (valid_color('#fff') -> format("  ✓ valid_color short hex~n") ; format("  ✗ valid_color short hex~n")),
    (\+ valid_color('invalid') -> format("  ✓ invalid color rejected~n") ; format("  ✗ invalid color accepted~n")),

    % Test CSS unit validation
    (valid_css_unit('10px') -> format("  ✓ valid_css_unit px~n") ; format("  ✗ valid_css_unit px~n")),
    (valid_css_unit('1.5rem') -> format("  ✓ valid_css_unit rem~n") ; format("  ✗ valid_css_unit rem~n")),
    (valid_css_unit('100%') -> format("  ✓ valid_css_unit percent~n") ; format("  ✗ valid_css_unit percent~n")),

    % Test identifier validation
    (valid_identifier(my_curve) -> format("  ✓ valid_identifier~n") ; format("  ✗ valid_identifier~n")),
    (\+ valid_identifier('123abc') -> format("  ✓ invalid identifier rejected~n") ; format("  ✗ invalid identifier accepted~n")),

    % Test error formatting
    format_error(error(error, test_code, "Test message", [key(value)]), Formatted),
    format("  ✓ Error formatted: ~w~n", [Formatted]),

    % Test validation report for missing spec
    validation_report(nonexistent_curve, curve, Report),
    format("  ✓ Validation report for missing: ~w~n", [Report]),

    format("Spec validation tests complete.~n").

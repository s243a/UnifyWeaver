% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Accessibility Generator - ARIA, Keyboard Navigation, and Focus Management
%
% This module provides declarative accessibility specifications that generate
% ARIA attributes, keyboard handlers, and focus management code.
%
% Usage:
%   % Define ARIA attributes for a component
%   aria_spec(chart_component, [
%       role(img),
%       label("Interactive chart showing sales data"),
%       describedby(chart_description)
%   ]).
%
%   % Define keyboard navigation
%   keyboard_nav(data_table, [
%       arrow_keys(navigate_cells),
%       enter(activate_cell),
%       escape(exit_edit_mode)
%   ]).
%
%   % Generate accessibility code
%   ?- generate_aria_props(chart_component, Props).

:- module(accessibility_generator, [
    % ARIA specifications
    aria_spec/2,                    % aria_spec(+Component, +Attributes)
    aria_role/2,                    % aria_role(+Component, +Role)

    % Keyboard navigation
    keyboard_nav/2,                 % keyboard_nav(+Component, +Handlers)
    focus_trap/2,                   % focus_trap(+Component, +Options)

    % Live region announcements
    live_region/2,                  % live_region(+Name, +Options)

    % Skip links
    skip_link/2,                    % skip_link(+Name, +Target)

    % Generation predicates
    generate_aria_props/2,          % generate_aria_props(+Component, -Props)
    generate_aria_jsx/2,            % generate_aria_jsx(+Component, -JSX)
    generate_keyboard_handler/2,    % generate_keyboard_handler(+Component, -Handler)
    generate_focus_trap_jsx/2,      % generate_focus_trap_jsx(+Component, -JSX)
    generate_skip_links_jsx/2,      % generate_skip_links_jsx(+Component, -JSX)
    generate_live_region_jsx/2,     % generate_live_region_jsx(+Name, -JSX)
    generate_accessibility_css/2,   % generate_accessibility_css(+Component, -CSS)

    % Utility predicates
    get_aria_label/2,               % get_aria_label(+Component, -Label)
    get_aria_role/2,                % get_aria_role(+Component, -Role)

    % Management
    declare_aria_spec/2,            % declare_aria_spec(+Component, +Attributes)
    declare_keyboard_nav/2,         % declare_keyboard_nav(+Component, +Handlers)
    clear_accessibility/0,          % clear_accessibility

    % Testing
    test_accessibility_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic aria_spec/2.
:- dynamic aria_role/2.
:- dynamic keyboard_nav/2.
:- dynamic focus_trap/2.
:- dynamic live_region/2.
:- dynamic skip_link/2.

:- discontiguous aria_spec/2.
:- discontiguous keyboard_nav/2.

% ============================================================================
% DEFAULT ARIA SPECIFICATIONS
% ============================================================================

% Chart components
aria_spec(line_chart, [
    role(img),
    label("Line chart visualization"),
    describedby(chart_description)
]).

aria_spec(bar_chart, [
    role(img),
    label("Bar chart visualization"),
    describedby(chart_description)
]).

aria_spec(pie_chart, [
    role(img),
    label("Pie chart visualization"),
    describedby(chart_description)
]).

aria_spec(scatter_plot, [
    role(img),
    label("Scatter plot visualization"),
    describedby(chart_description)
]).

aria_spec(heatmap, [
    role(img),
    label("Heatmap visualization"),
    describedby(chart_description)
]).

aria_spec(surface_3d, [
    role(img),
    label("3D surface plot"),
    describedby(chart_description)
]).

% Interactive components
aria_spec(data_table, [
    role(grid),
    label("Data table"),
    multiselectable(false),
    readonly(true)
]).

aria_spec(control_panel, [
    role(form),
    label("Chart controls")
]).

aria_spec(slider_control, [
    role(slider),
    valuemin(0),
    valuemax(100),
    valuenow(50)
]).

aria_spec(color_picker, [
    role(listbox),
    label("Color selection")
]).

% Navigation components
aria_spec(sidebar, [
    role(navigation),
    label("Sidebar navigation")
]).

aria_spec(main_content, [
    role(main),
    label("Main content area")
]).

aria_spec(chart_legend, [
    role(list),
    label("Chart legend")
]).

% ============================================================================
% DEFAULT KEYBOARD NAVIGATION
% ============================================================================

% Data table navigation
keyboard_nav(data_table, [
    key('ArrowUp', 'moveFocus("up")'),
    key('ArrowDown', 'moveFocus("down")'),
    key('ArrowLeft', 'moveFocus("left")'),
    key('ArrowRight', 'moveFocus("right")'),
    key('Home', 'moveFocus("first")'),
    key('End', 'moveFocus("last")'),
    key('Enter', 'activateCell()'),
    key('Escape', 'exitEditMode()'),
    key('Tab', 'moveFocus("next")')
]).

% Chart navigation (when interactive)
keyboard_nav(interactive_chart, [
    key('ArrowLeft', 'selectPreviousPoint()'),
    key('ArrowRight', 'selectNextPoint()'),
    key('ArrowUp', 'selectPreviousSeries()'),
    key('ArrowDown', 'selectNextSeries()'),
    key('Enter', 'showPointDetails()'),
    key('Escape', 'clearSelection()'),
    key(' ', 'togglePointSelection()')  % Space
]).

% Slider control
keyboard_nav(slider, [
    key('ArrowLeft', 'decrementValue(1)'),
    key('ArrowRight', 'incrementValue(1)'),
    key('ArrowUp', 'incrementValue(1)'),
    key('ArrowDown', 'decrementValue(1)'),
    key('PageUp', 'incrementValue(10)'),
    key('PageDown', 'decrementValue(10)'),
    key('Home', 'setMinValue()'),
    key('End', 'setMaxValue()')
]).

% Tab/panel navigation
keyboard_nav(tablist, [
    key('ArrowLeft', 'selectPreviousTab()'),
    key('ArrowRight', 'selectNextTab()'),
    key('Home', 'selectFirstTab()'),
    key('End', 'selectLastTab()'),
    key('Enter', 'activateTab()'),
    key(' ', 'activateTab()')
]).

% Modal/dialog navigation
keyboard_nav(modal, [
    key('Escape', 'closeModal()'),
    key('Tab', 'trapFocus()')
]).

% ============================================================================
% DEFAULT FOCUS TRAPS
% ============================================================================

focus_trap(modal_dialog, [
    container_selector('.modal'),
    initial_focus('.modal-close'),
    return_focus(true)
]).

focus_trap(dropdown_menu, [
    container_selector('.dropdown-menu'),
    initial_focus('.dropdown-item:first-child'),
    escape_closes(true)
]).

focus_trap(sidebar_expanded, [
    container_selector('.sidebar'),
    escape_closes(true),
    outside_click_closes(true)
]).

% ============================================================================
% DEFAULT LIVE REGIONS
% ============================================================================

live_region(chart_updates, [
    aria_live(polite),
    aria_atomic(true),
    role(status)
]).

live_region(error_messages, [
    aria_live(assertive),
    aria_atomic(true),
    role(alert)
]).

live_region(loading_status, [
    aria_live(polite),
    aria_busy(true),
    role(status)
]).

% ============================================================================
% DEFAULT SKIP LINKS
% ============================================================================

skip_link(main, '#main-content').
skip_link(nav, '#navigation').
skip_link(chart, '#chart-container').
skip_link(controls, '#control-panel').

% ============================================================================
% ARIA PROPS GENERATION
% ============================================================================

%% generate_aria_props(+Component, -Props)
%  Generate ARIA attributes as a props object string.
generate_aria_props(Component, Props) :-
    aria_spec(Component, Attributes),
    findall(PropStr, (
        member(Attr, Attributes),
        aria_attr_to_prop(Attr, PropStr)
    ), PropStrs),
    atomic_list_concat(PropStrs, ',\n  ', PropsInner),
    format(atom(Props), '{\n  ~w\n}', [PropsInner]).

%% aria_attr_to_prop(+Attr, -PropStr)
aria_attr_to_prop(role(Role), PropStr) :-
    format(atom(PropStr), 'role: "~w"', [Role]).
aria_attr_to_prop(label(Label), PropStr) :-
    format(atom(PropStr), '"aria-label": "~w"', [Label]).
aria_attr_to_prop(describedby(Id), PropStr) :-
    format(atom(PropStr), '"aria-describedby": "~w"', [Id]).
aria_attr_to_prop(labelledby(Id), PropStr) :-
    format(atom(PropStr), '"aria-labelledby": "~w"', [Id]).
aria_attr_to_prop(controls(Id), PropStr) :-
    format(atom(PropStr), '"aria-controls": "~w"', [Id]).
aria_attr_to_prop(expanded(Bool), PropStr) :-
    format(atom(PropStr), '"aria-expanded": ~w', [Bool]).
aria_attr_to_prop(selected(Bool), PropStr) :-
    format(atom(PropStr), '"aria-selected": ~w', [Bool]).
aria_attr_to_prop(hidden(Bool), PropStr) :-
    format(atom(PropStr), '"aria-hidden": ~w', [Bool]).
aria_attr_to_prop(disabled(Bool), PropStr) :-
    format(atom(PropStr), '"aria-disabled": ~w', [Bool]).
aria_attr_to_prop(readonly(Bool), PropStr) :-
    format(atom(PropStr), '"aria-readonly": ~w', [Bool]).
aria_attr_to_prop(multiselectable(Bool), PropStr) :-
    format(atom(PropStr), '"aria-multiselectable": ~w', [Bool]).
aria_attr_to_prop(valuemin(Val), PropStr) :-
    format(atom(PropStr), '"aria-valuemin": ~w', [Val]).
aria_attr_to_prop(valuemax(Val), PropStr) :-
    format(atom(PropStr), '"aria-valuemax": ~w', [Val]).
aria_attr_to_prop(valuenow(Val), PropStr) :-
    format(atom(PropStr), '"aria-valuenow": ~w', [Val]).
aria_attr_to_prop(valuetext(Text), PropStr) :-
    format(atom(PropStr), '"aria-valuetext": "~w"', [Text]).
aria_attr_to_prop(haspopup(Type), PropStr) :-
    format(atom(PropStr), '"aria-haspopup": "~w"', [Type]).
aria_attr_to_prop(live(Mode), PropStr) :-
    format(atom(PropStr), '"aria-live": "~w"', [Mode]).
aria_attr_to_prop(atomic(Bool), PropStr) :-
    format(atom(PropStr), '"aria-atomic": ~w', [Bool]).
aria_attr_to_prop(busy(Bool), PropStr) :-
    format(atom(PropStr), '"aria-busy": ~w', [Bool]).
aria_attr_to_prop(current(Value), PropStr) :-
    format(atom(PropStr), '"aria-current": "~w"', [Value]).
aria_attr_to_prop(level(N), PropStr) :-
    format(atom(PropStr), '"aria-level": ~w', [N]).
aria_attr_to_prop(posinset(N), PropStr) :-
    format(atom(PropStr), '"aria-posinset": ~w', [N]).
aria_attr_to_prop(setsize(N), PropStr) :-
    format(atom(PropStr), '"aria-setsize": ~w', [N]).
aria_attr_to_prop(sort(Order), PropStr) :-
    format(atom(PropStr), '"aria-sort": "~w"', [Order]).

%% generate_aria_jsx(+Component, -JSX)
%  Generate JSX spread props for ARIA attributes.
generate_aria_jsx(Component, JSX) :-
    generate_aria_props(Component, Props),
    format(atom(JSX), '{...~w}', [Props]).

% ============================================================================
% KEYBOARD HANDLER GENERATION
% ============================================================================

%% generate_keyboard_handler(+Component, -Handler)
%  Generate a keyboard event handler function.
generate_keyboard_handler(Component, Handler) :-
    keyboard_nav(Component, KeyMappings),
    findall(CaseStr, (
        member(key(Key, Action), KeyMappings),
        format(atom(CaseStr), '      case "~w":\n        ~w;\n        event.preventDefault();\n        break;', [Key, Action])
    ), CaseStrs),
    atomic_list_concat(CaseStrs, '\n', CasesCode),
    format(atom(Handler),
'const handleKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
~w
      default:
        break;
    }
  };', [CasesCode]).

%% generate_keyboard_hook(+Component, -Hook)
%  Generate a useEffect hook for keyboard handling.
generate_keyboard_hook(Component, Hook) :-
    generate_keyboard_handler(Component, Handler),
    format(atom(Hook),
'useEffect(() => {
    ~w

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);', [Handler]).

% ============================================================================
% FOCUS TRAP GENERATION
% ============================================================================

%% generate_focus_trap_jsx(+Component, -JSX)
%  Generate a focus trap component wrapper.
generate_focus_trap_jsx(Component, JSX) :-
    focus_trap(Component, Options),
    (member(container_selector(Selector), Options) -> true ; Selector = '.focus-trap'),
    (member(initial_focus(InitFocus), Options) -> true ; InitFocus = ':first-focusable'),
    (member(return_focus(Return), Options) -> true ; Return = true),
    (member(escape_closes(Escape), Options) -> true ; Escape = false),

    format(atom(JSX),
'<FocusTrap
  containerSelector="~w"
  initialFocus="~w"
  returnFocusOnDeactivate={~w}
  escapeDeactivates={~w}
>
  {children}
</FocusTrap>', [Selector, InitFocus, Return, Escape]).

%% generate_focus_trap_hook(-Hook)
%  Generate a custom useFocusTrap hook.
generate_focus_trap_hook(Hook) :-
    Hook =
'const useFocusTrap = (containerRef: React.RefObject<HTMLElement>, options: FocusTrapOptions = {}) => {
  const { initialFocus, returnFocusOnDeactivate = true, escapeDeactivates = true } = options;
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Store previous focus
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Get focusable elements
    const focusableElements = container.querySelectorAll(
      \'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])\'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    // Set initial focus
    if (initialFocus) {
      const initial = container.querySelector(initialFocus) as HTMLElement;
      initial?.focus();
    } else {
      firstElement?.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Tab") {
        if (event.shiftKey && document.activeElement === firstElement) {
          event.preventDefault();
          lastElement?.focus();
        } else if (!event.shiftKey && document.activeElement === lastElement) {
          event.preventDefault();
          firstElement?.focus();
        }
      }
      if (escapeDeactivates && event.key === "Escape") {
        if (returnFocusOnDeactivate) {
          previousFocusRef.current?.focus();
        }
      }
    };

    container.addEventListener("keydown", handleKeyDown);
    return () => {
      container.removeEventListener("keydown", handleKeyDown);
      if (returnFocusOnDeactivate) {
        previousFocusRef.current?.focus();
      }
    };
  }, [containerRef, initialFocus, returnFocusOnDeactivate, escapeDeactivates]);
};'.

% ============================================================================
% SKIP LINKS GENERATION
% ============================================================================

%% generate_skip_links_jsx(+Components, -JSX)
%  Generate skip links for accessibility.
generate_skip_links_jsx(Components, JSX) :-
    findall(LinkJSX, (
        member(Comp, Components),
        skip_link(Comp, Target),
        atom_string(Comp, CompStr),
        capitalize_first(CompStr, Label),
        format(atom(LinkJSX), '      <a href="~w" className={styles.skipLink}>Skip to ~w</a>', [Target, Label])
    ), LinkJSXList),
    atomic_list_concat(LinkJSXList, '\n', LinksCode),
    format(atom(JSX),
'<div className={styles.skipLinks}>
~w
    </div>', [LinksCode]).

%% generate_skip_links_css(-CSS)
generate_skip_links_css(CSS) :-
    CSS =
'.skipLinks {
  position: absolute;
  top: -100%;
  left: 0;
  z-index: 9999;
}

.skipLink {
  position: absolute;
  top: -100%;
  left: 0;
  padding: 0.5rem 1rem;
  background: var(--background, #1a1a2e);
  color: var(--text, #e0e0e0);
  text-decoration: none;
  border-radius: 0 0 4px 0;
  transition: top 0.2s ease;
}

.skipLink:focus {
  top: 0;
  outline: 2px solid var(--accent, #00d4ff);
  outline-offset: 2px;
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .skipLink {
    transition: none;
  }
}'.

% ============================================================================
% LIVE REGION GENERATION
% ============================================================================

%% generate_live_region_jsx(+Name, -JSX)
%  Generate a live region component.
generate_live_region_jsx(Name, JSX) :-
    live_region(Name, Options),
    (member(aria_live(Live), Options) -> true ; Live = polite),
    (member(aria_atomic(Atomic), Options) -> true ; Atomic = true),
    (member(role(Role), Options) -> true ; Role = status),

    atom_string(Name, NameStr),
    to_camel_case(NameStr, CamelName),

    format(atom(JSX),
'<div
  id="~w"
  role="~w"
  aria-live="~w"
  aria-atomic={~w}
  className={styles.srOnly}
>
  {~wMessage}
</div>', [Name, Role, Live, Atomic, CamelName]).

% ============================================================================
% ACCESSIBILITY CSS GENERATION
% ============================================================================

%% generate_accessibility_css(+Component, -CSS)
%  Generate accessibility-related CSS.
generate_accessibility_css(_Component, CSS) :-
    CSS =
'/* Screen reader only - visually hidden but accessible */
.srOnly {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Focus visible styles */
.focusVisible:focus-visible {
  outline: 2px solid var(--accent, #00d4ff);
  outline-offset: 2px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .focusVisible:focus-visible {
    outline: 3px solid currentColor;
    outline-offset: 3px;
  }
}

/* Focus within for composite widgets */
.focusWithin:focus-within {
  outline: 2px solid var(--accent, #00d4ff);
  outline-offset: 2px;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Touch target sizing */
.touchTarget {
  min-width: 44px;
  min-height: 44px;
}

/* Interactive element base styles */
.interactive {
  cursor: pointer;
  user-select: none;
}

.interactive:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* Loading state indicator */
[aria-busy="true"] {
  cursor: wait;
}

/* Hidden but maintains layout */
[aria-hidden="true"] {
  visibility: hidden;
}

/* Current/active item indicator */
[aria-current="true"],
[aria-current="page"] {
  font-weight: bold;
  border-left: 3px solid var(--accent, #00d4ff);
  padding-left: 0.5rem;
}'.

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_aria_label(+Component, -Label)
get_aria_label(Component, Label) :-
    aria_spec(Component, Attributes),
    member(label(Label), Attributes).

%% get_aria_role(+Component, -Role)
get_aria_role(Component, Role) :-
    aria_spec(Component, Attributes),
    member(role(Role), Attributes).

%% to_camel_case(+String, -CamelCase)
to_camel_case(String, CamelCase) :-
    atom_codes(String, Codes),
    camel_codes(Codes, true, CamelCodes),
    atom_codes(CamelCase, CamelCodes).

camel_codes([], _, []).
camel_codes([0'_|Rest], _, Result) :-
    camel_codes(Rest, true, Result).
camel_codes([C|Rest], true, [Upper|Result]) :-
    C \= 0'_,
    to_upper(C, Upper),
    camel_codes(Rest, false, Result).
camel_codes([C|Rest], false, [C|Result]) :-
    C \= 0'_,
    camel_codes(Rest, false, Result).

to_upper(C, U) :-
    C >= 0'a, C =< 0'z,
    !,
    U is C - 32.
to_upper(C, C).

%% capitalize_first(+String, -Capitalized)
capitalize_first(String, Capitalized) :-
    atom_codes(String, [First|Rest]),
    to_upper(First, Upper),
    atom_codes(Capitalized, [Upper|Rest]).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_aria_spec(+Component, +Attributes)
declare_aria_spec(Component, Attributes) :-
    retractall(aria_spec(Component, _)),
    assertz(aria_spec(Component, Attributes)).

%% declare_keyboard_nav(+Component, +Handlers)
declare_keyboard_nav(Component, Handlers) :-
    retractall(keyboard_nav(Component, _)),
    assertz(keyboard_nav(Component, Handlers)).

%% clear_accessibility
clear_accessibility :-
    retractall(aria_spec(_, _)),
    retractall(keyboard_nav(_, _)),
    retractall(focus_trap(_, _)),
    retractall(live_region(_, _)),
    retractall(skip_link(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_accessibility_generator :-
    format('~n========================================~n'),
    format('Accessibility Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: ARIA props generation
    format('Test 1: ARIA props generation~n'),
    generate_aria_props(line_chart, Props),
    (sub_atom(Props, _, _, _, 'role: "img"')
    -> format('  PASS: Props contain role~n')
    ; format('  FAIL: Props missing role~n')),
    (sub_atom(Props, _, _, _, 'aria-label')
    -> format('  PASS: Props contain aria-label~n')
    ; format('  FAIL: Props missing aria-label~n')),

    % Test 2: Keyboard handler generation
    format('~nTest 2: Keyboard handler generation~n'),
    generate_keyboard_handler(data_table, Handler),
    (sub_atom(Handler, _, _, _, 'handleKeyDown')
    -> format('  PASS: Handler has function name~n')
    ; format('  FAIL: Handler missing function name~n')),
    (sub_atom(Handler, _, _, _, 'ArrowUp')
    -> format('  PASS: Handler has ArrowUp case~n')
    ; format('  FAIL: Handler missing ArrowUp case~n')),
    (sub_atom(Handler, _, _, _, 'preventDefault')
    -> format('  PASS: Handler prevents default~n')
    ; format('  FAIL: Handler missing preventDefault~n')),

    % Test 3: Focus trap JSX
    format('~nTest 3: Focus trap generation~n'),
    generate_focus_trap_jsx(modal_dialog, FocusTrapJSX),
    (sub_atom(FocusTrapJSX, _, _, _, 'FocusTrap')
    -> format('  PASS: JSX has FocusTrap component~n')
    ; format('  FAIL: JSX missing FocusTrap component~n')),

    % Test 4: Skip links
    format('~nTest 4: Skip links generation~n'),
    generate_skip_links_jsx([main, nav, chart], SkipLinksJSX),
    (sub_atom(SkipLinksJSX, _, _, _, 'skipLink')
    -> format('  PASS: Has skip link class~n')
    ; format('  FAIL: Missing skip link class~n')),
    (sub_atom(SkipLinksJSX, _, _, _, '#main-content')
    -> format('  PASS: Has main content target~n')
    ; format('  FAIL: Missing main content target~n')),

    % Test 5: Live region
    format('~nTest 5: Live region generation~n'),
    generate_live_region_jsx(chart_updates, LiveRegionJSX),
    (sub_atom(LiveRegionJSX, _, _, _, 'aria-live')
    -> format('  PASS: Has aria-live attribute~n')
    ; format('  FAIL: Missing aria-live attribute~n')),
    (sub_atom(LiveRegionJSX, _, _, _, 'role="status"')
    -> format('  PASS: Has status role~n')
    ; format('  FAIL: Missing status role~n')),

    % Test 6: Accessibility CSS
    format('~nTest 6: Accessibility CSS generation~n'),
    generate_accessibility_css(_, AccessCSS),
    (sub_atom(AccessCSS, _, _, _, '.srOnly')
    -> format('  PASS: Has screen reader only class~n')
    ; format('  FAIL: Missing screen reader only class~n')),
    (sub_atom(AccessCSS, _, _, _, 'prefers-reduced-motion')
    -> format('  PASS: Has reduced motion support~n')
    ; format('  FAIL: Missing reduced motion support~n')),

    % Test 7: ARIA label query
    format('~nTest 7: ARIA label query~n'),
    get_aria_label(line_chart, Label),
    (Label \= ''
    -> format('  PASS: Got label: ~w~n', [Label])
    ; format('  FAIL: No label returned~n')),

    format('~nAll tests completed.~n').

:- initialization(test_accessibility_generator, main).

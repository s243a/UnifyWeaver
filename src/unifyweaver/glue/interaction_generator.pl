% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Interaction Generator - Event Handling, Tooltips, and Interactive Controls
%
% This module provides declarative specifications for chart/visualization
% interactions including events, tooltips, zoom, pan, and drag operations.
%
% Usage:
%   % Define interactions for a chart
%   interaction(line_chart, [
%       on_hover(show_tooltip),
%       on_click(select_point),
%       on_drag(pan_view),
%       on_scroll(zoom)
%   ]).
%
%   % Generate React event handlers
%   ?- generate_event_handlers(line_chart, Handlers).

:- module(interaction_generator, [
    % Interaction specifications
    interaction/2,                  % interaction(+Component, +Events)
    tooltip_spec/2,                 % tooltip_spec(+Component, +Options)
    zoom_spec/2,                    % zoom_spec(+Component, +Options)
    pan_spec/2,                     % pan_spec(+Component, +Options)
    drag_spec/2,                    % drag_spec(+Component, +Options)
    selection_spec/2,               % selection_spec(+Component, +Options)

    % Generation predicates
    generate_event_handlers/2,      % generate_event_handlers(+Component, -Handlers)
    generate_tooltip_jsx/2,         % generate_tooltip_jsx(+Component, -JSX)
    generate_tooltip_css/2,         % generate_tooltip_css(+Component, -CSS)
    generate_zoom_controls/2,       % generate_zoom_controls(+Component, -Controls)
    generate_pan_handler/2,         % generate_pan_handler(+Component, -Handler)
    generate_drag_handler/2,        % generate_drag_handler(+Component, -Handler)
    generate_selection_handler/2,   % generate_selection_handler(+Component, -Handler)

    % State management
    generate_interaction_state/2,   % generate_interaction_state(+Component, -State)
    generate_interaction_hooks/2,   % generate_interaction_hooks(+Component, -Hooks)

    % Utility predicates
    has_interaction/2,              % has_interaction(+Component, +Type)
    get_interaction_options/3,      % get_interaction_options(+Component, +Type, -Options)

    % Management
    declare_interaction/2,          % declare_interaction(+Component, +Events)
    declare_tooltip/2,              % declare_tooltip(+Component, +Options)
    clear_interactions/0,           % clear_interactions

    % Testing
    test_interaction_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic interaction/2.
:- dynamic tooltip_spec/2.
:- dynamic zoom_spec/2.
:- dynamic pan_spec/2.
:- dynamic drag_spec/2.
:- dynamic selection_spec/2.

:- discontiguous interaction/2.
:- discontiguous tooltip_spec/2.

% ============================================================================
% DEFAULT INTERACTION SPECIFICATIONS
% ============================================================================

% Line chart interactions
interaction(line_chart, [
    on_hover(show_tooltip),
    on_click(select_point),
    on_mouseleave(hide_tooltip)
]).

% Bar chart interactions
interaction(bar_chart, [
    on_hover(highlight_bar),
    on_click(select_bar),
    on_mouseleave(reset_highlight)
]).

% Scatter plot interactions
interaction(scatter_plot, [
    on_hover(show_tooltip),
    on_click(select_point),
    on_brush(select_range),
    on_scroll(zoom),
    on_drag(pan)
]).

% Pie chart interactions
interaction(pie_chart, [
    on_hover(expand_slice),
    on_click(select_slice),
    on_mouseleave(collapse_slice)
]).

% Heatmap interactions
interaction(heatmap, [
    on_hover(show_cell_value),
    on_click(select_cell),
    on_scroll(zoom)
]).

% Graph/Network interactions
interaction(network_graph, [
    on_node_hover(show_node_info),
    on_node_click(select_node),
    on_edge_hover(show_edge_info),
    on_drag(move_node),
    on_scroll(zoom),
    on_background_drag(pan)
]).

% 3D plot interactions
interaction(plot3d, [
    on_drag(rotate_view),
    on_scroll(zoom),
    on_hover(show_tooltip),
    on_double_click(reset_view)
]).

% Data table interactions
interaction(data_table, [
    on_header_click(sort_column),
    on_row_hover(highlight_row),
    on_row_click(select_row),
    on_cell_double_click(edit_cell)
]).

% ============================================================================
% DEFAULT TOOLTIP SPECIFICATIONS
% ============================================================================

tooltip_spec(default, [
    position(cursor),
    offset(10, 10),
    delay(200),
    animation(fade_in),
    style([
        background('rgba(0, 0, 0, 0.85)'),
        color('#ffffff'),
        padding('8px 12px'),
        border_radius('4px'),
        font_size('12px'),
        box_shadow('0 2px 8px rgba(0, 0, 0, 0.3)'),
        pointer_events('none'),
        z_index(1000)
    ])
]).

tooltip_spec(line_chart, [
    position(data_point),
    offset(0, -10),
    delay(100),
    animation(fade_in),
    content([
        field(x, "X"),
        field(y, "Y"),
        field(series, "Series")
    ]),
    style([
        background('var(--surface, #16213e)'),
        color('var(--text, #e0e0e0)'),
        border('1px solid var(--accent, #00d4ff)')
    ])
]).

tooltip_spec(bar_chart, [
    position(top_center),
    offset(0, -8),
    delay(100),
    content([
        field(label, "Category"),
        field(value, "Value")
    ])
]).

tooltip_spec(scatter_plot, [
    position(cursor),
    offset(12, 12),
    delay(150),
    content([
        field(x, "X"),
        field(y, "Y"),
        field(label, "Label")
    ])
]).

tooltip_spec(heatmap, [
    position(cursor),
    offset(15, 15),
    content([
        field(row, "Row"),
        field(col, "Column"),
        field(value, "Value")
    ])
]).

% ============================================================================
% ZOOM SPECIFICATIONS
% ============================================================================

zoom_spec(default, [
    enabled(true),
    min_scale(0.1),
    max_scale(10),
    step(0.1),
    scroll_sensitivity(0.001),
    pinch_enabled(true),
    double_click_reset(true),
    controls([zoom_in, zoom_out, reset])
]).

zoom_spec(scatter_plot, [
    enabled(true),
    min_scale(0.5),
    max_scale(20),
    step(0.2),
    axis_lock(none),
    controls([zoom_in, zoom_out, zoom_fit, reset])
]).

zoom_spec(heatmap, [
    enabled(true),
    min_scale(1),
    max_scale(5),
    constrain_to_data(true)
]).

% ============================================================================
% PAN SPECIFICATIONS
% ============================================================================

pan_spec(default, [
    enabled(true),
    mode(free),
    inertia(true),
    inertia_decay(0.95),
    bounds(none)
]).

pan_spec(scatter_plot, [
    enabled(true),
    mode(free),
    inertia(true),
    bounds(data_extent)
]).

pan_spec(network_graph, [
    enabled(true),
    mode(free),
    inertia(false),
    bounds(none)
]).

% ============================================================================
% DRAG SPECIFICATIONS
% ============================================================================

drag_spec(default, [
    enabled(true),
    threshold(5),
    handle(element),
    constrain(none)
]).

drag_spec(network_graph, [
    enabled(true),
    mode(node_move),
    snap_to_grid(false),
    update_layout(true)
]).

drag_spec(plot3d, [
    enabled(true),
    mode(rotate),
    sensitivity(0.5),
    inertia(true)
]).

% ============================================================================
% SELECTION SPECIFICATIONS
% ============================================================================

selection_spec(default, [
    mode(single),
    highlight(true),
    persistent(false)
]).

selection_spec(scatter_plot, [
    mode(multiple),
    method(brush),
    brush_type(rectangle),
    highlight(true)
]).

selection_spec(data_table, [
    mode(multiple),
    method(click),
    shift_extend(true),
    ctrl_toggle(true)
]).

% ============================================================================
% EVENT HANDLER GENERATION
% ============================================================================

%% generate_event_handlers(+Component, -Handlers)
%  Generate React event handlers for a component.
generate_event_handlers(Component, Handlers) :-
    (interaction(Component, Events) -> true ; Events = []),
    maplist(generate_single_handler, Events, HandlerList),
    atomic_list_concat(HandlerList, '\n\n', Handlers).

%% generate_single_handler(+Event, -Handler)
generate_single_handler(on_hover(Action), Handler) :-
    format(atom(Handler), 'const handleMouseEnter = (event: React.MouseEvent, data: DataPoint) => {
  ~w(data, event);
};', [Action]).

generate_single_handler(on_click(Action), Handler) :-
    format(atom(Handler), 'const handleClick = (event: React.MouseEvent, data: DataPoint) => {
  event.preventDefault();
  ~w(data, event);
};', [Action]).

generate_single_handler(on_mouseleave(Action), Handler) :-
    format(atom(Handler), 'const handleMouseLeave = (event: React.MouseEvent) => {
  ~w(event);
};', [Action]).

generate_single_handler(on_scroll(zoom), Handler) :-
    Handler = 'const handleWheel = (event: React.WheelEvent) => {
  event.preventDefault();
  const delta = event.deltaY > 0 ? -0.1 : 0.1;
  setScale(prev => Math.max(minScale, Math.min(maxScale, prev + delta)));
};'.

generate_single_handler(on_drag(pan), Handler) :-
    Handler = 'const handleDrag = useCallback((event: MouseEvent) => {
  if (!isDragging) return;
  const dx = event.clientX - dragStart.x;
  const dy = event.clientY - dragStart.y;
  setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
  setDragStart({ x: event.clientX, y: event.clientY });
}, [isDragging, dragStart]);'.

generate_single_handler(on_brush(select_range), Handler) :-
    Handler = 'const handleBrush = (selection: BrushSelection) => {
  const selectedPoints = data.filter(point =>
    point.x >= selection.x0 && point.x <= selection.x1 &&
    point.y >= selection.y0 && point.y <= selection.y1
  );
  setSelectedData(selectedPoints);
  onSelectionChange?.(selectedPoints);
};'.

generate_single_handler(on_node_hover(Action), Handler) :-
    format(atom(Handler), 'const handleNodeHover = (node: Node, event: React.MouseEvent) => {
  ~w(node, event);
};', [Action]).

generate_single_handler(on_node_click(Action), Handler) :-
    format(atom(Handler), 'const handleNodeClick = (node: Node, event: React.MouseEvent) => {
  event.stopPropagation();
  ~w(node, event);
};', [Action]).

generate_single_handler(on_edge_hover(Action), Handler) :-
    format(atom(Handler), 'const handleEdgeHover = (edge: Edge, event: React.MouseEvent) => {
  ~w(edge, event);
};', [Action]).

generate_single_handler(on_background_drag(pan), Handler) :-
    Handler = 'const handleBackgroundDrag = useCallback((event: MouseEvent) => {
  if (!isPanning) return;
  const dx = event.clientX - panStart.x;
  const dy = event.clientY - panStart.y;
  setViewOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
  setPanStart({ x: event.clientX, y: event.clientY });
}, [isPanning, panStart]);'.

generate_single_handler(on_drag(rotate_view), Handler) :-
    Handler = 'const handleRotate = useCallback((event: MouseEvent) => {
  if (!isRotating) return;
  const dx = event.clientX - rotateStart.x;
  const dy = event.clientY - rotateStart.y;
  setRotation(prev => ({
    x: prev.x + dy * 0.5,
    y: prev.y + dx * 0.5
  }));
  setRotateStart({ x: event.clientX, y: event.clientY });
}, [isRotating, rotateStart]);'.

generate_single_handler(on_double_click(reset_view), Handler) :-
    Handler = 'const handleDoubleClick = () => {
  setScale(1);
  setOffset({ x: 0, y: 0 });
  setRotation({ x: 0, y: 0 });
};'.

generate_single_handler(on_header_click(sort_column), Handler) :-
    Handler = 'const handleHeaderClick = (column: string) => {
  setSortConfig(prev => ({
    column,
    direction: prev.column === column && prev.direction === "asc" ? "desc" : "asc"
  }));
};'.

generate_single_handler(on_row_hover(highlight_row), Handler) :-
    Handler = 'const handleRowHover = (rowIndex: number) => {
  setHighlightedRow(rowIndex);
};'.

generate_single_handler(on_row_click(select_row), Handler) :-
    Handler = 'const handleRowClick = (rowIndex: number, event: React.MouseEvent) => {
  if (event.shiftKey && lastSelectedRow !== null) {
    // Range selection
    const start = Math.min(lastSelectedRow, rowIndex);
    const end = Math.max(lastSelectedRow, rowIndex);
    const range = Array.from({ length: end - start + 1 }, (_, i) => start + i);
    setSelectedRows(prev => [...new Set([...prev, ...range])]);
  } else if (event.ctrlKey || event.metaKey) {
    // Toggle selection
    setSelectedRows(prev =>
      prev.includes(rowIndex)
        ? prev.filter(r => r !== rowIndex)
        : [...prev, rowIndex]
    );
  } else {
    // Single selection
    setSelectedRows([rowIndex]);
  }
  setLastSelectedRow(rowIndex);
};'.

generate_single_handler(on_cell_double_click(edit_cell), Handler) :-
    Handler = 'const handleCellDoubleClick = (rowIndex: number, column: string) => {
  setEditingCell({ row: rowIndex, column });
};'.

generate_single_handler(highlight_bar, Handler) :-
    Handler = 'const highlightBar = (barIndex: number) => {
  setHighlightedBar(barIndex);
};'.

generate_single_handler(expand_slice, Handler) :-
    Handler = 'const expandSlice = (sliceIndex: number) => {
  setExpandedSlice(sliceIndex);
};'.

generate_single_handler(on_drag(move_node), Handler) :-
    Handler = 'const handleNodeDrag = useCallback((nodeId: string, event: MouseEvent) => {
  if (!isDraggingNode) return;
  const newX = event.clientX - dragOffset.x;
  const newY = event.clientY - dragOffset.y;
  updateNodePosition(nodeId, { x: newX, y: newY });
}, [isDraggingNode, dragOffset, updateNodePosition]);'.

% Default handler for unknown events
generate_single_handler(Event, Handler) :-
    Event =.. [EventType, Action],
    format(atom(Handler), '// TODO: Implement handler for ~w(~w)', [EventType, Action]).

% ============================================================================
% TOOLTIP GENERATION
% ============================================================================

%% generate_tooltip_jsx(+Component, -JSX)
generate_tooltip_jsx(Component, JSX) :-
    (tooltip_spec(Component, Spec) -> true ; tooltip_spec(default, Spec)),
    (member(content(Fields), Spec) -> true ; Fields = []),
    generate_tooltip_content(Fields, ContentJSX),
    format(atom(JSX), 'interface TooltipProps {
  visible: boolean;
  x: number;
  y: number;
  data: Record<string, unknown>;
}

export const Tooltip: React.FC<TooltipProps> = ({ visible, x, y, data }) => {
  if (!visible) return null;

  return (
    <div
      className={styles.tooltip}
      style={{
        position: "absolute",
        left: x,
        top: y,
        transform: "translate(-50%, -100%)"
      }}
    >
~w
    </div>
  );
};', [ContentJSX]).

%% generate_tooltip_content(+Fields, -JSX)
generate_tooltip_content([], '      <span>{JSON.stringify(data)}</span>').
generate_tooltip_content(Fields, JSX) :-
    Fields \= [],
    maplist(generate_tooltip_field, Fields, FieldJSXs),
    atomic_list_concat(FieldJSXs, '\n', JSX).

%% generate_tooltip_field(+Field, -JSX)
generate_tooltip_field(field(Key, Label), JSX) :-
    format(atom(JSX), '      <div className={styles.tooltipRow}>\n        <span className={styles.tooltipLabel}>~w:</span>\n        <span className={styles.tooltipValue}>{data.~w}</span>\n      </div>', [Label, Key]).

%% generate_tooltip_css(+Component, -CSS)
generate_tooltip_css(Component, CSS) :-
    (tooltip_spec(Component, Spec) -> true ; tooltip_spec(default, Spec)),
    (member(style(Styles), Spec) -> true ; Styles = []),
    generate_style_rules(Styles, StyleRules),
    format(atom(CSS), '.tooltip {\n  position: absolute;\n  pointer-events: none;\n  z-index: 1000;\n~w}

.tooltipRow {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 2px 0;
}

.tooltipLabel {
  color: var(--text-secondary, #888);
  font-weight: 500;
}

.tooltipValue {
  font-family: monospace;
}', [StyleRules]).

%% generate_style_rules(+Styles, -CSS)
generate_style_rules(Styles, CSS) :-
    maplist(generate_style_rule, Styles, Rules),
    atomic_list_concat(Rules, CSS).

%% generate_style_rule(+Style, -CSS)
generate_style_rule(Style, CSS) :-
    Style =.. [PropName, Value],
    css_property_name(PropName, CSSName),
    format(atom(CSS), '  ~w: ~w;\n', [CSSName, Value]).

css_property_name(background, 'background').
css_property_name(color, 'color').
css_property_name(padding, 'padding').
css_property_name(border_radius, 'border-radius').
css_property_name(font_size, 'font-size').
css_property_name(box_shadow, 'box-shadow').
css_property_name(pointer_events, 'pointer-events').
css_property_name(z_index, 'z-index').
css_property_name(border, 'border').
css_property_name(Name, Name) :- atom(Name).

% ============================================================================
% ZOOM CONTROLS GENERATION
% ============================================================================

%% generate_zoom_controls(+Component, -Controls)
generate_zoom_controls(Component, Controls) :-
    (zoom_spec(Component, Spec) -> true ; zoom_spec(default, Spec)),
    (member(controls(ControlList), Spec) -> true ; ControlList = [zoom_in, zoom_out, reset]),
    generate_zoom_control_jsx(ControlList, Controls).

%% generate_zoom_control_jsx(+ControlList, -JSX)
generate_zoom_control_jsx(ControlList, JSX) :-
    maplist(generate_single_zoom_control, ControlList, ControlJSXs),
    atomic_list_concat(ControlJSXs, '\n      ', ControlsBody),
    format(atom(JSX), 'interface ZoomControlsProps {
  scale: number;
  minScale: number;
  maxScale: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onZoomFit?: () => void;
}

export const ZoomControls: React.FC<ZoomControlsProps> = ({
  scale,
  minScale,
  maxScale,
  onZoomIn,
  onZoomOut,
  onReset,
  onZoomFit
}) => {
  return (
    <div className={styles.zoomControls}>
      ~w
      <span className={styles.zoomLevel}>{Math.round(scale * 100)}%%</span>
    </div>
  );
};', [ControlsBody]).

generate_single_zoom_control(zoom_in, '<button onClick={onZoomIn} disabled={scale >= maxScale} title="Zoom In">+</button>').
generate_single_zoom_control(zoom_out, '<button onClick={onZoomOut} disabled={scale <= minScale} title="Zoom Out">-</button>').
generate_single_zoom_control(reset, '<button onClick={onReset} title="Reset Zoom">Reset</button>').
generate_single_zoom_control(zoom_fit, '<button onClick={onZoomFit} title="Fit to View">Fit</button>').

% ============================================================================
% PAN HANDLER GENERATION
% ============================================================================

%% generate_pan_handler(+Component, -Handler)
generate_pan_handler(Component, Handler) :-
    (pan_spec(Component, Spec) -> true ; pan_spec(default, Spec)),
    (member(inertia(true), Spec) -> InertiaCode = generate_inertia_code ; InertiaCode = ''),
    format(atom(Handler), 'const usePan = () => {
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const velocityRef = useRef({ x: 0, y: 0 });

  const handlePanStart = useCallback((event: React.MouseEvent) => {
    setIsPanning(true);
    setPanStart({ x: event.clientX, y: event.clientY });
  }, []);

  const handlePanMove = useCallback((event: MouseEvent) => {
    if (!isPanning) return;
    const dx = event.clientX - panStart.x;
    const dy = event.clientY - panStart.y;
    velocityRef.current = { x: dx, y: dy };
    setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
    setPanStart({ x: event.clientX, y: event.clientY });
  }, [isPanning, panStart]);

  const handlePanEnd = useCallback(() => {
    setIsPanning(false);
    ~w
  }, []);

  useEffect(() => {
    if (isPanning) {
      window.addEventListener("mousemove", handlePanMove);
      window.addEventListener("mouseup", handlePanEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handlePanMove);
      window.removeEventListener("mouseup", handlePanEnd);
    };
  }, [isPanning, handlePanMove, handlePanEnd]);

  return { offset, handlePanStart, isPanning };
};', [InertiaCode]).

generate_inertia_code('// Apply inertia
    const decay = 0.95;
    const animate = () => {
      if (Math.abs(velocityRef.current.x) > 0.1 || Math.abs(velocityRef.current.y) > 0.1) {
        velocityRef.current.x *= decay;
        velocityRef.current.y *= decay;
        setOffset(prev => ({
          x: prev.x + velocityRef.current.x,
          y: prev.y + velocityRef.current.y
        }));
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);').

% ============================================================================
% DRAG HANDLER GENERATION
% ============================================================================

%% generate_drag_handler(+Component, -Handler)
generate_drag_handler(Component, Handler) :-
    (drag_spec(Component, Spec) -> true ; drag_spec(default, Spec)),
    (member(mode(Mode), Spec) -> true ; Mode = free),
    generate_drag_handler_for_mode(Mode, Handler).

generate_drag_handler_for_mode(free, Handler) :-
    Handler = 'const useDrag = (elementRef: React.RefObject<HTMLElement>) => {
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const dragStartRef = useRef({ x: 0, y: 0 });

  const handleDragStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    setIsDragging(true);
    dragStartRef.current = {
      x: event.clientX - position.x,
      y: event.clientY - position.y
    };
  }, [position]);

  const handleDragMove = useCallback((event: MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: event.clientX - dragStartRef.current.x,
      y: event.clientY - dragStartRef.current.y
    });
  }, [isDragging]);

  const handleDragEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleDragMove);
      window.addEventListener("mouseup", handleDragEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handleDragMove);
      window.removeEventListener("mouseup", handleDragEnd);
    };
  }, [isDragging, handleDragMove, handleDragEnd]);

  return { position, isDragging, handleDragStart };
};'.

generate_drag_handler_for_mode(rotate, Handler) :-
    Handler = 'const useRotate = () => {
  const [isRotating, setIsRotating] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const rotateStartRef = useRef({ x: 0, y: 0 });

  const handleRotateStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    setIsRotating(true);
    rotateStartRef.current = { x: event.clientX, y: event.clientY };
  }, []);

  const handleRotateMove = useCallback((event: MouseEvent) => {
    if (!isRotating) return;
    const dx = event.clientX - rotateStartRef.current.x;
    const dy = event.clientY - rotateStartRef.current.y;
    setRotation(prev => ({
      x: prev.x + dy * 0.5,
      y: prev.y + dx * 0.5,
      z: prev.z
    }));
    rotateStartRef.current = { x: event.clientX, y: event.clientY };
  }, [isRotating]);

  const handleRotateEnd = useCallback(() => {
    setIsRotating(false);
  }, []);

  useEffect(() => {
    if (isRotating) {
      window.addEventListener("mousemove", handleRotateMove);
      window.addEventListener("mouseup", handleRotateEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handleRotateMove);
      window.removeEventListener("mouseup", handleRotateEnd);
    };
  }, [isRotating, handleRotateMove, handleRotateEnd]);

  return { rotation, isRotating, handleRotateStart, setRotation };
};'.

generate_drag_handler_for_mode(node_move, Handler) :-
    Handler = 'const useNodeDrag = (onNodeMove: (id: string, pos: Position) => void) => {
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const dragOffsetRef = useRef({ x: 0, y: 0 });

  const handleNodeDragStart = useCallback((nodeId: string, event: React.MouseEvent, nodePos: Position) => {
    event.stopPropagation();
    setDraggedNode(nodeId);
    dragOffsetRef.current = {
      x: event.clientX - nodePos.x,
      y: event.clientY - nodePos.y
    };
  }, []);

  const handleNodeDragMove = useCallback((event: MouseEvent) => {
    if (!draggedNode) return;
    const newPos = {
      x: event.clientX - dragOffsetRef.current.x,
      y: event.clientY - dragOffsetRef.current.y
    };
    onNodeMove(draggedNode, newPos);
  }, [draggedNode, onNodeMove]);

  const handleNodeDragEnd = useCallback(() => {
    setDraggedNode(null);
  }, []);

  useEffect(() => {
    if (draggedNode) {
      window.addEventListener("mousemove", handleNodeDragMove);
      window.addEventListener("mouseup", handleNodeDragEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handleNodeDragMove);
      window.removeEventListener("mouseup", handleNodeDragEnd);
    };
  }, [draggedNode, handleNodeDragMove, handleNodeDragEnd]);

  return { draggedNode, handleNodeDragStart };
};'.

% ============================================================================
% SELECTION HANDLER GENERATION
% ============================================================================

%% generate_selection_handler(+Component, -Handler)
generate_selection_handler(Component, Handler) :-
    (selection_spec(Component, Spec) -> true ; selection_spec(default, Spec)),
    (member(mode(Mode), Spec) -> true ; Mode = single),
    (member(method(Method), Spec) -> true ; Method = click),
    generate_selection_handler_for_mode(Mode, Method, Handler).

generate_selection_handler_for_mode(single, click, Handler) :-
    Handler = 'const useSelection = <T,>(items: T[]) => {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const handleSelect = useCallback((index: number) => {
    setSelectedIndex(prev => prev === index ? null : index);
  }, []);

  const selectedItem = selectedIndex !== null ? items[selectedIndex] : null;

  return { selectedIndex, selectedItem, handleSelect };
};'.

generate_selection_handler_for_mode(multiple, click, Handler) :-
    Handler = 'const useMultiSelect = <T,>(items: T[]) => {
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const lastSelectedRef = useRef<number | null>(null);

  const handleSelect = useCallback((index: number, event: React.MouseEvent) => {
    if (event.shiftKey && lastSelectedRef.current !== null) {
      // Range selection
      const start = Math.min(lastSelectedRef.current, index);
      const end = Math.max(lastSelectedRef.current, index);
      const range = new Set(Array.from({ length: end - start + 1 }, (_, i) => start + i));
      setSelectedIndices(prev => new Set([...prev, ...range]));
    } else if (event.ctrlKey || event.metaKey) {
      // Toggle selection
      setSelectedIndices(prev => {
        const next = new Set(prev);
        if (next.has(index)) next.delete(index);
        else next.add(index);
        return next;
      });
    } else {
      // Single selection
      setSelectedIndices(new Set([index]));
    }
    lastSelectedRef.current = index;
  }, []);

  const selectedItems = Array.from(selectedIndices).map(i => items[i]);

  return { selectedIndices, selectedItems, handleSelect };
};'.

generate_selection_handler_for_mode(multiple, brush, Handler) :-
    Handler = 'const useBrushSelection = <T extends { x: number; y: number },>(items: T[]) => {
  const [brushExtent, setBrushExtent] = useState<BrushExtent | null>(null);
  const [isBrushing, setIsBrushing] = useState(false);
  const brushStartRef = useRef({ x: 0, y: 0 });

  const handleBrushStart = useCallback((event: React.MouseEvent) => {
    setIsBrushing(true);
    const rect = event.currentTarget.getBoundingClientRect();
    brushStartRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    setBrushExtent(null);
  }, []);

  const handleBrushMove = useCallback((event: MouseEvent) => {
    if (!isBrushing) return;
    const rect = (event.target as Element).getBoundingClientRect();
    const currentX = event.clientX - rect.left;
    const currentY = event.clientY - rect.top;
    setBrushExtent({
      x0: Math.min(brushStartRef.current.x, currentX),
      y0: Math.min(brushStartRef.current.y, currentY),
      x1: Math.max(brushStartRef.current.x, currentX),
      y1: Math.max(brushStartRef.current.y, currentY)
    });
  }, [isBrushing]);

  const handleBrushEnd = useCallback(() => {
    setIsBrushing(false);
  }, []);

  const selectedItems = brushExtent
    ? items.filter(item =>
        item.x >= brushExtent.x0 && item.x <= brushExtent.x1 &&
        item.y >= brushExtent.y0 && item.y <= brushExtent.y1
      )
    : [];

  useEffect(() => {
    if (isBrushing) {
      window.addEventListener("mousemove", handleBrushMove);
      window.addEventListener("mouseup", handleBrushEnd);
    }
    return () => {
      window.removeEventListener("mousemove", handleBrushMove);
      window.removeEventListener("mouseup", handleBrushEnd);
    };
  }, [isBrushing, handleBrushMove, handleBrushEnd]);

  return { brushExtent, selectedItems, handleBrushStart, isBrushing };
};'.

% ============================================================================
% STATE AND HOOKS GENERATION
% ============================================================================

%% generate_interaction_state(+Component, -State)
generate_interaction_state(Component, State) :-
    (interaction(Component, _) -> HasInteraction = true ; HasInteraction = false),
    (tooltip_spec(Component, _) -> HasTooltip = true ; tooltip_spec(default, _), HasTooltip = true),
    (zoom_spec(Component, _) -> HasZoom = true ; HasZoom = false),
    generate_state_declarations(HasInteraction, HasTooltip, HasZoom, State).

generate_state_declarations(true, true, true, State) :-
    State = '// Tooltip state
const [tooltipVisible, setTooltipVisible] = useState(false);
const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
const [tooltipData, setTooltipData] = useState<Record<string, unknown>>({});

// Zoom/Pan state
const [scale, setScale] = useState(1);
const [offset, setOffset] = useState({ x: 0, y: 0 });

// Selection state
const [selectedData, setSelectedData] = useState<DataPoint[]>([]);
const [hoveredData, setHoveredData] = useState<DataPoint | null>(null);'.

generate_state_declarations(true, true, false, State) :-
    State = '// Tooltip state
const [tooltipVisible, setTooltipVisible] = useState(false);
const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
const [tooltipData, setTooltipData] = useState<Record<string, unknown>>({});

// Selection state
const [selectedData, setSelectedData] = useState<DataPoint[]>([]);
const [hoveredData, setHoveredData] = useState<DataPoint | null>(null);'.

generate_state_declarations(true, false, _, State) :-
    State = '// Selection state
const [selectedData, setSelectedData] = useState<DataPoint[]>([]);
const [hoveredData, setHoveredData] = useState<DataPoint | null>(null);'.

generate_state_declarations(false, _, _, State) :-
    State = '// No interaction state needed'.

%% generate_interaction_hooks(+Component, -Hooks)
generate_interaction_hooks(Component, Hooks) :-
    (interaction(Component, Events) -> true ; Events = []),
    findall(Hook, (member(Event, Events), event_requires_hook(Event, Hook)), HookList),
    sort(HookList, UniqueHooks),
    maplist(atom_string, UniqueHooks, _HookStrs),
    format(atom(Hooks), 'import { useState, useCallback, useEffect, useRef } from "react";', []).

event_requires_hook(on_drag(_), useCallback).
event_requires_hook(on_scroll(_), useCallback).
event_requires_hook(on_brush(_), useCallback).
event_requires_hook(on_background_drag(_), useCallback).
event_requires_hook(_, useState).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% has_interaction(+Component, +Type)
has_interaction(Component, Type) :-
    interaction(Component, Events),
    member(Event, Events),
    Event =.. [Type|_].

%% get_interaction_options(+Component, +Type, -Options)
get_interaction_options(Component, tooltip, Options) :-
    (tooltip_spec(Component, Options) -> true ; tooltip_spec(default, Options)).
get_interaction_options(Component, zoom, Options) :-
    (zoom_spec(Component, Options) -> true ; zoom_spec(default, Options)).
get_interaction_options(Component, pan, Options) :-
    (pan_spec(Component, Options) -> true ; pan_spec(default, Options)).
get_interaction_options(Component, drag, Options) :-
    (drag_spec(Component, Options) -> true ; drag_spec(default, Options)).
get_interaction_options(Component, selection, Options) :-
    (selection_spec(Component, Options) -> true ; selection_spec(default, Options)).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_interaction(+Component, +Events)
declare_interaction(Component, Events) :-
    retractall(interaction(Component, _)),
    assertz(interaction(Component, Events)).

%% declare_tooltip(+Component, +Options)
declare_tooltip(Component, Options) :-
    retractall(tooltip_spec(Component, _)),
    assertz(tooltip_spec(Component, Options)).

%% clear_interactions/0
clear_interactions :-
    retractall(interaction(_, _)),
    retractall(tooltip_spec(_, _)),
    retractall(zoom_spec(_, _)),
    retractall(pan_spec(_, _)),
    retractall(drag_spec(_, _)),
    retractall(selection_spec(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_interaction_generator :-
    format('========================================~n'),
    format('Interaction Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: Event handlers generation
    format('Test 1: Event handlers generation~n'),
    generate_event_handlers(line_chart, Handlers),
    (sub_atom(Handlers, _, _, _, 'handleMouseEnter')
    -> format('  PASS: Has mouse enter handler~n')
    ; format('  FAIL: Missing mouse enter handler~n')),
    (sub_atom(Handlers, _, _, _, 'handleClick')
    -> format('  PASS: Has click handler~n')
    ; format('  FAIL: Missing click handler~n')),

    % Test 2: Tooltip generation
    format('~nTest 2: Tooltip generation~n'),
    generate_tooltip_jsx(line_chart, TooltipJSX),
    (sub_atom(TooltipJSX, _, _, _, 'Tooltip')
    -> format('  PASS: Has Tooltip component~n')
    ; format('  FAIL: Missing Tooltip component~n')),
    generate_tooltip_css(line_chart, TooltipCSS),
    (sub_atom(TooltipCSS, _, _, _, '.tooltip')
    -> format('  PASS: Has tooltip class~n')
    ; format('  FAIL: Missing tooltip class~n')),

    % Test 3: Zoom controls generation
    format('~nTest 3: Zoom controls generation~n'),
    generate_zoom_controls(scatter_plot, ZoomControls),
    (sub_atom(ZoomControls, _, _, _, 'ZoomControls')
    -> format('  PASS: Has ZoomControls component~n')
    ; format('  FAIL: Missing ZoomControls~n')),
    (sub_atom(ZoomControls, _, _, _, 'onZoomIn')
    -> format('  PASS: Has zoom in handler~n')
    ; format('  FAIL: Missing zoom in~n')),

    % Test 4: Pan handler generation
    format('~nTest 4: Pan handler generation~n'),
    generate_pan_handler(scatter_plot, PanHandler),
    (sub_atom(PanHandler, _, _, _, 'usePan')
    -> format('  PASS: Has usePan hook~n')
    ; format('  FAIL: Missing usePan~n')),
    (sub_atom(PanHandler, _, _, _, 'handlePanStart')
    -> format('  PASS: Has pan start handler~n')
    ; format('  FAIL: Missing pan start~n')),

    % Test 5: Drag handler generation
    format('~nTest 5: Drag handler generation~n'),
    generate_drag_handler(plot3d, DragHandler),
    (sub_atom(DragHandler, _, _, _, 'useRotate')
    -> format('  PASS: Has useRotate hook~n')
    ; format('  FAIL: Missing useRotate~n')),
    generate_drag_handler(network_graph, NodeDragHandler),
    (sub_atom(NodeDragHandler, _, _, _, 'useNodeDrag')
    -> format('  PASS: Has useNodeDrag hook~n')
    ; format('  FAIL: Missing useNodeDrag~n')),

    % Test 6: Selection handler generation
    format('~nTest 6: Selection handler generation~n'),
    generate_selection_handler(data_table, SelectionHandler),
    (sub_atom(SelectionHandler, _, _, _, 'useMultiSelect')
    -> format('  PASS: Has useMultiSelect hook~n')
    ; format('  FAIL: Missing useMultiSelect~n')),
    generate_selection_handler(scatter_plot, BrushHandler),
    (sub_atom(BrushHandler, _, _, _, 'useBrushSelection')
    -> format('  PASS: Has useBrushSelection hook~n')
    ; format('  FAIL: Missing useBrushSelection~n')),

    % Test 7: Interaction state generation
    format('~nTest 7: Interaction state generation~n'),
    generate_interaction_state(scatter_plot, State),
    (sub_atom(State, _, _, _, 'tooltipVisible')
    -> format('  PASS: Has tooltip state~n')
    ; format('  FAIL: Missing tooltip state~n')),
    (sub_atom(State, _, _, _, 'scale')
    -> format('  PASS: Has zoom scale state~n')
    ; format('  FAIL: Missing zoom scale~n')),

    % Test 8: Utility predicates
    format('~nTest 8: Utility predicates~n'),
    (has_interaction(line_chart, on_hover)
    -> format('  PASS: line_chart has on_hover~n')
    ; format('  FAIL: line_chart missing on_hover~n')),
    get_interaction_options(scatter_plot, zoom, ZoomOpts),
    (member(enabled(true), ZoomOpts)
    -> format('  PASS: scatter_plot zoom enabled~n')
    ; format('  FAIL: scatter_plot zoom not enabled~n')),

    format('~nAll tests completed.~n').

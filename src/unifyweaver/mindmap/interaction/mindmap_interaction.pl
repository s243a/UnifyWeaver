% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_interaction.pl - Mind Map Interaction Event Model
%
% Defines interaction specifications for mind map visualizations.
% Extends the core interaction_generator.pl with mind map-specific modes.
%
% Usage:
%   ?- mindmap_interaction(my_map, [on_node_click(follow_link)]).
%   ?- generate_mindmap_interaction(my_map, Code).

:- module(mindmap_interaction, [
    % Interaction specifications
    mindmap_interaction/2,
    mindmap_tooltip_spec/2,
    mindmap_selection_spec/2,
    mindmap_drag_spec/2,
    mindmap_gesture_spec/2,

    % Generation predicates
    generate_mindmap_interaction/2,
    generate_mindmap_handlers/2,
    generate_mindmap_events_jsx/2,
    generate_gesture_handlers/2,

    % Management
    declare_mindmap_interaction/2,
    declare_mindmap_gesture_spec/2,
    clear_mindmap_interactions/0,

    % Queries
    has_mindmap_interaction/2,
    get_mindmap_event_handler/3,
    get_gesture_spec/2,

    % Testing
    test_mindmap_interaction/0,
    test_gesture_support/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic mindmap_interaction/2.
:- dynamic mindmap_tooltip_spec/2.
:- dynamic mindmap_selection_spec/2.
:- dynamic mindmap_drag_spec/2.
:- dynamic mindmap_gesture_spec/2.

% ============================================================================
% DEFAULT INTERACTION SPECIFICATIONS
% ============================================================================

% Standard mind map interactions
mindmap_interaction(default, [
    on_node_hover(show_tooltip),
    on_node_click(select_node),
    on_node_double_click(follow_link),
    on_node_drag(move_node),
    on_edge_hover(highlight_edge),
    on_background_click(deselect_all),
    on_scroll(zoom),
    on_background_drag(pan)
]).

% Read-only mode (no editing)
mindmap_interaction(read_only, [
    on_node_hover(show_tooltip),
    on_node_click(follow_link),
    on_scroll(zoom),
    on_background_drag(pan)
]).

% Edit mode (full editing capabilities)
mindmap_interaction(edit_mode, [
    on_node_hover(show_tooltip),
    on_node_click(select_node),
    on_node_double_click(edit_label),
    on_node_drag(move_node),
    on_edge_click(select_edge),
    on_background_click(deselect_all),
    on_background_double_click(create_node),
    on_scroll(zoom),
    on_background_drag(pan),
    on_keypress(keyboard_shortcuts)
]).

% Presentation mode (simplified interactions)
mindmap_interaction(presentation, [
    on_node_click(focus_branch),
    on_scroll(navigate_branches),
    on_keypress(presentation_controls)
]).

% ============================================================================
% DEFAULT TOOLTIP SPECIFICATIONS
% ============================================================================

mindmap_tooltip_spec(default, [
    position(cursor),
    offset(10, 10),
    delay(200),
    show_on(hover),
    content([label, type, link_preview]),
    style([
        background('rgba(0, 0, 0, 0.85)'),
        color('#ffffff'),
        padding('8px 12px'),
        border_radius('4px'),
        font_size('12px'),
        max_width('200px')
    ])
]).

mindmap_tooltip_spec(rich, [
    position(node_right),
    offset(20, 0),
    delay(300),
    show_on(hover),
    content([label, description, link, children_count]),
    style([
        background('#ffffff'),
        color('#333333'),
        padding('12px 16px'),
        border_radius('8px'),
        box_shadow('0 4px 12px rgba(0,0,0,0.15)'),
        font_size('13px'),
        max_width('300px')
    ])
]).

% ============================================================================
% DEFAULT SELECTION SPECIFICATIONS
% ============================================================================

mindmap_selection_spec(default, [
    mode(single),
    method(click),
    highlight_style([
        stroke('#ff6b6b'),
        stroke_width(3),
        stroke_dasharray('none')
    ]),
    on_select(highlight_branch)
]).

mindmap_selection_spec(multi_select, [
    mode(multiple),
    method(click_with_modifier),
    modifier_key(ctrl),
    highlight_style([
        stroke('#4ecdc4'),
        stroke_width(2),
        stroke_dasharray('5,3')
    ]),
    on_select(add_to_selection),
    on_deselect(remove_from_selection)
]).

mindmap_selection_spec(brush_select, [
    mode(range),
    method(brush),
    brush_style([
        fill('rgba(78, 205, 196, 0.2)'),
        stroke('#4ecdc4')
    ]),
    on_select(select_in_range)
]).

% ============================================================================
% DEFAULT DRAG SPECIFICATIONS
% ============================================================================

mindmap_drag_spec(default, [
    enabled(true),
    mode(node_move),
    constrain(none),
    snap_to_grid(false),
    update_layout(true),
    on_drag_start(pause_simulation),
    on_drag(update_position),
    on_drag_end(resume_simulation)
]).

mindmap_drag_spec(constrained, [
    enabled(true),
    mode(node_move),
    constrain(bounds),
    bounds([0, 0, 1600, 1200]),
    snap_to_grid(true),
    grid_size(20),
    update_layout(false),
    on_drag(snap_position)
]).

mindmap_drag_spec(branch_move, [
    enabled(true),
    mode(branch_move),
    include_children(true),
    maintain_relative(true),
    on_drag(move_branch)
]).

% ============================================================================
% TOUCH GESTURE SPECIFICATIONS
% ============================================================================

%% mindmap_gesture_spec(+Id, -Spec)
%
%  Gesture specifications for touch device support.
%
%  Supported gestures:
%  - tap: Single finger tap (maps to click)
%  - double_tap: Quick double tap (maps to double-click)
%  - long_press: Hold for duration (context menu/selection)
%  - pinch: Two-finger pinch (zoom in/out)
%  - pan: Single/two finger drag (scroll/pan)
%  - rotate: Two-finger rotate (optional rotation)
%  - swipe: Quick directional swipe (navigation)
%

mindmap_gesture_spec(default, [
    % Tap gestures
    gesture(tap, [
        fingers(1),
        max_duration(300),
        action(select_node),
        fallback_action(deselect_all)  % When tapping background
    ]),
    gesture(double_tap, [
        fingers(1),
        interval(300),
        action(follow_link),
        fallback_action(zoom_to_fit)
    ]),
    gesture(long_press, [
        fingers(1),
        duration(500),
        action(show_context_menu),
        visual_feedback(pulse)
    ]),

    % Multi-touch gestures
    gesture(pinch, [
        fingers(2),
        action(zoom),
        min_scale(0.25),
        max_scale(4.0),
        inertia(true)
    ]),
    gesture(pan, [
        fingers(1),
        action(scroll),
        inertia(true),
        inertia_deceleration(0.95)
    ]),
    gesture(two_finger_pan, [
        fingers(2),
        action(pan_viewport),
        inertia(true)
    ]),

    % Swipe gestures
    gesture(swipe_left, [
        fingers(1),
        min_velocity(0.5),
        min_distance(50),
        action(next_sibling)
    ]),
    gesture(swipe_right, [
        fingers(1),
        min_velocity(0.5),
        min_distance(50),
        action(previous_sibling)
    ]),
    gesture(swipe_up, [
        fingers(1),
        min_velocity(0.5),
        min_distance(50),
        action(go_to_parent)
    ]),
    gesture(swipe_down, [
        fingers(1),
        min_velocity(0.5),
        min_distance(50),
        action(expand_children)
    ]),

    % Configuration
    config([
        prevent_default(true),
        touch_action(none),
        passive_listeners(false)
    ])
]).

mindmap_gesture_spec(read_only, [
    gesture(tap, [
        fingers(1),
        max_duration(300),
        action(follow_link)
    ]),
    gesture(double_tap, [
        fingers(1),
        interval(300),
        action(zoom_to_node)
    ]),
    gesture(pinch, [
        fingers(2),
        action(zoom),
        min_scale(0.5),
        max_scale(3.0)
    ]),
    gesture(pan, [
        fingers(1),
        action(scroll)
    ]),
    config([
        prevent_default(true),
        passive_listeners(true)
    ])
]).

mindmap_gesture_spec(edit_mode, [
    gesture(tap, [
        fingers(1),
        max_duration(300),
        action(select_node)
    ]),
    gesture(double_tap, [
        fingers(1),
        interval(300),
        action(edit_label)
    ]),
    gesture(long_press, [
        fingers(1),
        duration(400),
        action(start_drag),
        visual_feedback(lift)
    ]),
    gesture(pinch, [
        fingers(2),
        action(zoom)
    ]),
    gesture(pan, [
        fingers(2),  % Two fingers for pan in edit mode
        action(pan_viewport)
    ]),
    gesture(drag, [
        fingers(1),
        requires(selected_node),
        action(move_node)
    ]),
    gesture(two_finger_tap, [
        fingers(2),
        action(undo)
    ]),
    gesture(three_finger_tap, [
        fingers(3),
        action(redo)
    ]),
    config([
        prevent_default(true),
        touch_action(none)
    ])
]).

mindmap_gesture_spec(presentation, [
    gesture(tap, [
        fingers(1),
        action(focus_branch)
    ]),
    gesture(swipe_left, [
        fingers(1),
        action(next_slide)
    ]),
    gesture(swipe_right, [
        fingers(1),
        action(previous_slide)
    ]),
    gesture(pinch_out, [
        fingers(2),
        action(zoom_out_overview)
    ]),
    gesture(pinch_in, [
        fingers(2),
        action(zoom_in_detail)
    ]),
    config([
        prevent_default(true),
        fullscreen_gestures(true)
    ])
]).

% ============================================================================
% GENERATION PREDICATES
% ============================================================================

%% generate_mindmap_interaction(+MapId, -Code)
%
%  Generate complete interaction code for a mind map.
%
generate_mindmap_interaction(MapId, Code) :-
    % Get interaction spec (use default if not defined)
    (   mindmap_interaction(MapId, Events)
    ->  true
    ;   mindmap_interaction(default, Events)
    ),

    % Get tooltip spec
    (   mindmap_tooltip_spec(MapId, TooltipOpts)
    ->  true
    ;   mindmap_tooltip_spec(default, TooltipOpts)
    ),

    % Get selection spec
    (   mindmap_selection_spec(MapId, SelectionOpts)
    ->  true
    ;   mindmap_selection_spec(default, SelectionOpts)
    ),

    % Get drag spec
    (   mindmap_drag_spec(MapId, DragOpts)
    ->  true
    ;   mindmap_drag_spec(default, DragOpts)
    ),

    % Generate code sections
    generate_event_handlers_code(Events, EventHandlersCode),
    generate_tooltip_code(TooltipOpts, TooltipCode),
    generate_selection_code(SelectionOpts, SelectionCode),
    generate_drag_code(DragOpts, DragCode),

    % Combine
    format(string(Code),
"// Mind Map Interactions for: ~w
// Generated by UnifyWeaver

~w

~w

~w

~w
", [MapId, EventHandlersCode, TooltipCode, SelectionCode, DragCode]).

%% generate_mindmap_handlers(+MapId, -Handlers)
%
%  Generate just the event handler functions.
%
generate_mindmap_handlers(MapId, Handlers) :-
    (   mindmap_interaction(MapId, Events)
    ->  true
    ;   mindmap_interaction(default, Events)
    ),
    generate_event_handlers_code(Events, Handlers).

%% generate_mindmap_events_jsx(+MapId, -JSX)
%
%  Generate JSX event bindings for React.
%
generate_mindmap_events_jsx(MapId, JSX) :-
    (   mindmap_interaction(MapId, Events)
    ->  true
    ;   mindmap_interaction(default, Events)
    ),
    generate_jsx_event_bindings(Events, JSX).

% ============================================================================
% CODE GENERATION HELPERS
% ============================================================================

generate_event_handlers_code(Events, Code) :-
    findall(HandlerCode, (member(Event, Events), event_to_handler(Event, HandlerCode)), Handlers),
    atomic_list_concat(Handlers, '\n\n', Code).

event_to_handler(on_node_hover(show_tooltip), Code) :-
    Code = "const handleNodeHover = (event, node) => {
    showTooltip(event, node);
};".

event_to_handler(on_node_click(select_node), Code) :-
    Code = "const handleNodeClick = (event, node) => {
    if (selectedNode !== node.id) {
        setSelectedNode(node.id);
        highlightBranch(node);
    }
};".

event_to_handler(on_node_click(follow_link), Code) :-
    Code = "const handleNodeClick = (event, node) => {
    if (node.url) {
        window.open(node.url, '_blank');
    }
};".

event_to_handler(on_node_double_click(follow_link), Code) :-
    Code = "const handleNodeDoubleClick = (event, node) => {
    if (node.url) {
        window.open(node.url, '_blank');
    }
};".

event_to_handler(on_node_double_click(edit_label), Code) :-
    Code = "const handleNodeDoubleClick = (event, node) => {
    setEditingNode(node.id);
    setEditText(node.label);
};".

event_to_handler(on_node_drag(move_node), Code) :-
    Code = "const handleNodeDrag = (event, node) => {
    node.fx = event.x;
    node.fy = event.y;
    updateNodePosition(node);
};".

event_to_handler(on_edge_hover(highlight_edge), Code) :-
    Code = "const handleEdgeHover = (event, edge) => {
    highlightEdge(edge);
};".

event_to_handler(on_background_click(deselect_all), Code) :-
    Code = "const handleBackgroundClick = () => {
    setSelectedNode(null);
    clearHighlights();
};".

event_to_handler(on_background_double_click(create_node), Code) :-
    Code = "const handleBackgroundDoubleClick = (event) => {
    const [x, y] = d3.pointer(event);
    createNewNode(x, y);
};".

event_to_handler(on_scroll(zoom), Code) :-
    Code = "const handleScroll = (event) => {
    // Handled by D3 zoom behavior
};".

event_to_handler(on_background_drag(pan), Code) :-
    Code = "const handleBackgroundDrag = (event) => {
    // Handled by D3 zoom behavior
};".

event_to_handler(on_keypress(keyboard_shortcuts), Code) :-
    Code = "const handleKeypress = (event) => {
    switch (event.key) {
        case 'Delete':
        case 'Backspace':
            if (selectedNode) deleteNode(selectedNode);
            break;
        case 'Escape':
            setSelectedNode(null);
            setEditingNode(null);
            break;
        case 'Enter':
            if (editingNode) commitEdit();
            break;
    }
};".

event_to_handler(on_node_click(focus_branch), Code) :-
    Code = "const handleNodeClick = (event, node) => {
    focusOnBranch(node);
};".

event_to_handler(on_scroll(navigate_branches), Code) :-
    Code = "const handleScroll = (event) => {
    if (event.deltaY > 0) nextBranch();
    else previousBranch();
};".

event_to_handler(on_keypress(presentation_controls), Code) :-
    Code = "const handleKeypress = (event) => {
    switch (event.key) {
        case 'ArrowRight':
        case 'Space':
            nextBranch();
            break;
        case 'ArrowLeft':
            previousBranch();
            break;
        case 'Home':
            goToRoot();
            break;
        case 'Escape':
            exitPresentation();
            break;
    }
};".

event_to_handler(on_edge_click(select_edge), Code) :-
    Code = "const handleEdgeClick = (event, edge) => {
    setSelectedEdge(edge);
    highlightEdge(edge);
};".

event_to_handler(_, "// Handler not implemented").

% Tooltip code generation
generate_tooltip_code(TooltipOpts, Code) :-
    member(position(Position), TooltipOpts),
    member(delay(Delay), TooltipOpts),
    format(string(Code),
"// Tooltip Setup
const tooltipConfig = {
    position: '~w',
    delay: ~w,
    show: (event, node) => {
        const tooltip = document.getElementById('mindmap-tooltip');
        tooltip.innerHTML = formatTooltipContent(node);
        tooltip.style.visibility = 'visible';
        positionTooltip(tooltip, event, '~w');
    },
    hide: () => {
        const tooltip = document.getElementById('mindmap-tooltip');
        tooltip.style.visibility = 'hidden';
    }
};

const showTooltip = (event, node) => {
    clearTimeout(tooltipTimer);
    tooltipTimer = setTimeout(() => tooltipConfig.show(event, node), ~w);
};

const hideTooltip = () => {
    clearTimeout(tooltipTimer);
    tooltipConfig.hide();
};

const formatTooltipContent = (node) => {
    let content = `<strong>${node.label}</strong>`;
    if (node.type) content += `<br/><small>Type: ${node.type}</small>`;
    if (node.url) content += `<br/><small>Click to open link</small>`;
    return content;
};
", [Position, Delay, Position, Delay]).

% Selection code generation
generate_selection_code(SelectionOpts, Code) :-
    member(mode(Mode), SelectionOpts),
    member(method(Method), SelectionOpts),
    format(string(Code),
"// Selection Setup
const selectionConfig = {
    mode: '~w',
    method: '~w'
};

let selectedNode = null;
let selectedNodes = [];

const selectNode = (node) => {
    if (selectionConfig.mode === 'single') {
        selectedNode = node.id;
        selectedNodes = [node.id];
    } else {
        if (!selectedNodes.includes(node.id)) {
            selectedNodes.push(node.id);
        }
    }
    updateSelectionVisuals();
};

const deselectNode = (node) => {
    selectedNodes = selectedNodes.filter(id => id !== node.id);
    if (selectedNode === node.id) selectedNode = null;
    updateSelectionVisuals();
};

const clearSelection = () => {
    selectedNode = null;
    selectedNodes = [];
    updateSelectionVisuals();
};

const highlightBranch = (node) => {
    // Highlight the node and its descendants
    const descendants = getDescendants(node.id);
    descendants.forEach(id => highlightNode(id));
};
", [Mode, Method]).

% Drag code generation
generate_drag_code(DragOpts, Code) :-
    member(enabled(Enabled), DragOpts),
    member(mode(Mode), DragOpts),
    (   member(snap_to_grid(true), DragOpts),
        member(grid_size(GridSize), DragOpts)
    ->  SnapCode = format(string(S), "
        x = Math.round(x / ~w) * ~w;
        y = Math.round(y / ~w) * ~w;", [GridSize, GridSize, GridSize, GridSize]), S
    ;   SnapCode = ""
    ),
    format(string(Code),
"// Drag Setup
const dragConfig = {
    enabled: ~w,
    mode: '~w'
};

const createDragBehavior = () => {
    if (!dragConfig.enabled) return null;

    return d3.drag()
        .on('start', (event, d) => {
            if (simulation) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        })
        .on('drag', (event, d) => {
            let x = event.x;
            let y = event.y;
~w
            d.fx = x;
            d.fy = y;
        })
        .on('end', (event, d) => {
            if (simulation) simulation.alphaTarget(0);
            // Keep position fixed after drag
        });
};
", [Enabled, Mode, SnapCode]).

% JSX event bindings
generate_jsx_event_bindings(Events, JSX) :-
    findall(Binding, (member(Event, Events), event_to_jsx_binding(Event, Binding)), Bindings),
    atomic_list_concat(Bindings, '\n', JSX).

event_to_jsx_binding(on_node_hover(_), "onMouseEnter={handleNodeHover}\nonMouseLeave={hideTooltip}").
event_to_jsx_binding(on_node_click(_), "onClick={handleNodeClick}").
event_to_jsx_binding(on_node_double_click(_), "onDoubleClick={handleNodeDoubleClick}").
event_to_jsx_binding(on_keypress(_), "onKeyDown={handleKeypress}").
event_to_jsx_binding(_, "").

% ============================================================================
% GESTURE CODE GENERATION
% ============================================================================

%% generate_gesture_handlers(+MapId, -Code)
%
%  Generate touch gesture handling code for a mind map.
%
generate_gesture_handlers(MapId, Code) :-
    (   mindmap_gesture_spec(MapId, GestureSpec)
    ->  true
    ;   mindmap_gesture_spec(default, GestureSpec)
    ),
    generate_gesture_config(GestureSpec, ConfigCode),
    generate_gesture_recognizers(GestureSpec, RecognizerCode),
    generate_gesture_actions(GestureSpec, ActionCode),
    format(string(Code),
"// Touch Gesture Support for: ~w
// Generated by UnifyWeaver

~w

~w

~w

// Initialize gesture handling
const initGestures = (element) => {
    const hammer = new Hammer(element, gestureConfig);
    setupGestureRecognizers(hammer);
    bindGestureActions(hammer);
    return hammer;
};
", [MapId, ConfigCode, RecognizerCode, ActionCode]).

%% generate_gesture_config(+GestureSpec, -Code)
generate_gesture_config(GestureSpec, Code) :-
    (   member(config(ConfigOpts), GestureSpec)
    ->  true
    ;   ConfigOpts = []
    ),
    (   member(prevent_default(PreventDefault), ConfigOpts)
    ->  true
    ;   PreventDefault = true
    ),
    (   member(touch_action(TouchAction), ConfigOpts)
    ->  true
    ;   TouchAction = 'auto'
    ),
    format(string(Code),
"// Gesture Configuration
const gestureConfig = {
    touchAction: '~w',
    preventDefault: ~w,
    recognizers: []
};

// Touch state tracking
let touchState = {
    startTime: 0,
    startPos: { x: 0, y: 0 },
    currentPos: { x: 0, y: 0 },
    fingers: 0,
    velocity: { x: 0, y: 0 },
    scale: 1,
    rotation: 0
};
", [TouchAction, PreventDefault]).

%% generate_gesture_recognizers(+GestureSpec, -Code)
generate_gesture_recognizers(GestureSpec, Code) :-
    findall(RecognizerCode,
        (   member(gesture(GestureType, Opts), GestureSpec),
            gesture_to_recognizer(GestureType, Opts, RecognizerCode)
        ),
        Recognizers),
    atomic_list_concat(Recognizers, '\n', RecognizersStr),
    format(string(Code),
"// Gesture Recognizers
const setupGestureRecognizers = (hammer) => {
~w
};
", [RecognizersStr]).

gesture_to_recognizer(tap, Opts, Code) :-
    (   member(max_duration(Duration), Opts)
    ->  true
    ;   Duration = 300
    ),
    format(string(Code),
"    hammer.get('tap').set({ time: ~w });", [Duration]).

gesture_to_recognizer(double_tap, Opts, Code) :-
    (   member(interval(Interval), Opts)
    ->  true
    ;   Interval = 300
    ),
    format(string(Code),
"    hammer.get('doubletap').set({ taps: 2, interval: ~w });", [Interval]).

gesture_to_recognizer(long_press, Opts, Code) :-
    (   member(duration(Duration), Opts)
    ->  true
    ;   Duration = 500
    ),
    format(string(Code),
"    hammer.get('press').set({ time: ~w });", [Duration]).

gesture_to_recognizer(pinch, Opts, Code) :-
    (   member(min_scale(MinScale), Opts)
    ->  true
    ;   MinScale = 0.25
    ),
    (   member(max_scale(MaxScale), Opts)
    ->  true
    ;   MaxScale = 4.0
    ),
    format(string(Code),
"    hammer.get('pinch').set({ enable: true, threshold: 0.1 });
    // Scale limits: ~w to ~w", [MinScale, MaxScale]).

gesture_to_recognizer(pan, Opts, Code) :-
    (   member(fingers(Fingers), Opts)
    ->  true
    ;   Fingers = 1
    ),
    format(string(Code),
"    hammer.get('pan').set({ direction: Hammer.DIRECTION_ALL, pointers: ~w });", [Fingers]).

gesture_to_recognizer(two_finger_pan, _, Code) :-
    Code = "    hammer.get('pan').set({ direction: Hammer.DIRECTION_ALL, pointers: 2 });".

gesture_to_recognizer(swipe_left, Opts, Code) :-
    gesture_to_swipe_recognizer(Opts, Code).
gesture_to_recognizer(swipe_right, Opts, Code) :-
    gesture_to_swipe_recognizer(Opts, Code).
gesture_to_recognizer(swipe_up, Opts, Code) :-
    gesture_to_swipe_recognizer(Opts, Code).
gesture_to_recognizer(swipe_down, Opts, Code) :-
    gesture_to_swipe_recognizer(Opts, Code).

gesture_to_swipe_recognizer(Opts, Code) :-
    (   member(min_velocity(Velocity), Opts)
    ->  true
    ;   Velocity = 0.5
    ),
    (   member(min_distance(Distance), Opts)
    ->  true
    ;   Distance = 50
    ),
    format(string(Code),
"    hammer.get('swipe').set({ velocity: ~w, distance: ~w });", [Velocity, Distance]).

gesture_to_recognizer(two_finger_tap, _, Code) :-
    Code = "    hammer.add(new Hammer.Tap({ event: 'twofingertap', pointers: 2 }));".

gesture_to_recognizer(three_finger_tap, _, Code) :-
    Code = "    hammer.add(new Hammer.Tap({ event: 'threefingertap', pointers: 3 }));".

gesture_to_recognizer(drag, _, Code) :-
    Code = "    // Drag handled via pan with selected node".

gesture_to_recognizer(pinch_in, _, Code) :-
    Code = "    hammer.get('pinch').set({ enable: true });".

gesture_to_recognizer(pinch_out, _, Code) :-
    Code = "    hammer.get('pinch').set({ enable: true });".

gesture_to_recognizer(_, _, "    // Unknown gesture type").

%% generate_gesture_actions(+GestureSpec, -Code)
generate_gesture_actions(GestureSpec, Code) :-
    findall(ActionCode,
        (   member(gesture(GestureType, Opts), GestureSpec),
            member(action(Action), Opts),
            gesture_to_action_binding(GestureType, Action, ActionCode)
        ),
        Actions),
    atomic_list_concat(Actions, '\n', ActionsStr),
    format(string(Code),
"// Gesture Action Bindings
const bindGestureActions = (hammer) => {
~w
};

// Gesture action implementations
const gestureActions = {
    select_node: (event) => {
        const target = event.target.closest('.node');
        if (target) selectNode(getNodeData(target));
    },
    deselect_all: () => {
        clearSelection();
    },
    follow_link: (event) => {
        const target = event.target.closest('.node');
        if (target) {
            const node = getNodeData(target);
            if (node.url) window.open(node.url, '_blank');
        }
    },
    zoom_to_fit: () => {
        zoomToFit();
    },
    zoom_to_node: (event) => {
        const target = event.target.closest('.node');
        if (target) zoomToNode(getNodeData(target));
    },
    show_context_menu: (event) => {
        showContextMenu(event.center.x, event.center.y, event.target);
    },
    zoom: (event) => {
        const scale = currentScale * event.scale;
        setZoom(Math.min(Math.max(scale, minScale), maxScale));
    },
    scroll: (event) => {
        panBy(event.deltaX, event.deltaY);
    },
    pan_viewport: (event) => {
        panBy(event.deltaX, event.deltaY);
    },
    next_sibling: () => navigateToSibling(1),
    previous_sibling: () => navigateToSibling(-1),
    go_to_parent: () => navigateToParent(),
    expand_children: () => expandChildren(),
    edit_label: (event) => {
        const target = event.target.closest('.node');
        if (target) startEditingNode(getNodeData(target));
    },
    start_drag: (event) => {
        const target = event.target.closest('.node');
        if (target) startDrag(getNodeData(target));
    },
    move_node: (event) => {
        if (draggedNode) moveNode(draggedNode, event.deltaX, event.deltaY);
    },
    undo: () => undoAction(),
    redo: () => redoAction(),
    focus_branch: (event) => {
        const target = event.target.closest('.node');
        if (target) focusOnBranch(getNodeData(target));
    },
    next_slide: () => nextSlide(),
    previous_slide: () => previousSlide(),
    zoom_out_overview: () => zoomToOverview(),
    zoom_in_detail: () => zoomToDetail()
};
", [ActionsStr]).

gesture_to_action_binding(tap, Action, Code) :-
    format(string(Code),
"    hammer.on('tap', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(double_tap, Action, Code) :-
    format(string(Code),
"    hammer.on('doubletap', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(long_press, Action, Code) :-
    format(string(Code),
"    hammer.on('press', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(pinch, Action, Code) :-
    format(string(Code),
"    hammer.on('pinch', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(pan, Action, Code) :-
    format(string(Code),
"    hammer.on('pan', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(two_finger_pan, Action, Code) :-
    format(string(Code),
"    hammer.on('pan', (e) => { if (e.pointers.length === 2) gestureActions.~w(e); });", [Action]).

gesture_to_action_binding(swipe_left, Action, Code) :-
    format(string(Code),
"    hammer.on('swipeleft', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(swipe_right, Action, Code) :-
    format(string(Code),
"    hammer.on('swiperight', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(swipe_up, Action, Code) :-
    format(string(Code),
"    hammer.on('swipeup', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(swipe_down, Action, Code) :-
    format(string(Code),
"    hammer.on('swipedown', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(two_finger_tap, Action, Code) :-
    format(string(Code),
"    hammer.on('twofingertap', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(three_finger_tap, Action, Code) :-
    format(string(Code),
"    hammer.on('threefingertap', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(drag, Action, Code) :-
    format(string(Code),
"    hammer.on('pan', (e) => { if (selectedNode) gestureActions.~w(e); });", [Action]).

gesture_to_action_binding(pinch_in, Action, Code) :-
    format(string(Code),
"    hammer.on('pinchin', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(pinch_out, Action, Code) :-
    format(string(Code),
"    hammer.on('pinchout', (e) => gestureActions.~w(e));", [Action]).

gesture_to_action_binding(_, _, "    // Unknown gesture binding").

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_mindmap_interaction(+MapId, +Events)
declare_mindmap_interaction(MapId, Events) :-
    retractall(mindmap_interaction(MapId, _)),
    assertz(mindmap_interaction(MapId, Events)).

%% declare_mindmap_gesture_spec(+MapId, +GestureSpec)
declare_mindmap_gesture_spec(MapId, GestureSpec) :-
    retractall(mindmap_gesture_spec(MapId, _)),
    assertz(mindmap_gesture_spec(MapId, GestureSpec)).

%% clear_mindmap_interactions
clear_mindmap_interactions :-
    retractall(mindmap_interaction(_, _)),
    retractall(mindmap_tooltip_spec(_, _)),
    retractall(mindmap_selection_spec(_, _)),
    retractall(mindmap_drag_spec(_, _)),
    retractall(mindmap_gesture_spec(_, _)).

%% has_mindmap_interaction(+MapId, +EventType)
has_mindmap_interaction(MapId, EventType) :-
    mindmap_interaction(MapId, Events),
    member(Event, Events),
    Event =.. [EventType, _].

%% get_mindmap_event_handler(+MapId, +EventType, -Handler)
get_mindmap_event_handler(MapId, EventType, Handler) :-
    mindmap_interaction(MapId, Events),
    member(Event, Events),
    Event =.. [EventType, Handler].

%% get_gesture_spec(+MapId, -GestureSpec)
get_gesture_spec(MapId, GestureSpec) :-
    (   mindmap_gesture_spec(MapId, GestureSpec)
    ->  true
    ;   mindmap_gesture_spec(default, GestureSpec)
    ).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_interaction :-
    format('~n=== Mind Map Interaction Tests ===~n~n'),

    % Test 1: Default interactions exist
    format('Test 1: Default interactions...~n'),
    (   mindmap_interaction(default, Events),
        member(on_node_hover(_), Events)
    ->  format('  PASS: Default interactions defined~n')
    ;   format('  FAIL: Default interactions missing~n')
    ),

    % Test 2: Generate interaction code
    format('~nTest 2: Generate interaction code...~n'),
    generate_mindmap_interaction(default, Code),
    (   sub_string(Code, _, _, _, "handleNodeClick")
    ->  format('  PASS: Interaction code generated~n')
    ;   format('  FAIL: Code generation failed~n')
    ),

    % Test 3: Has interaction check
    format('~nTest 3: Has interaction check...~n'),
    (   has_mindmap_interaction(default, on_scroll)
    ->  format('  PASS: Has on_scroll interaction~n')
    ;   format('  FAIL: Missing on_scroll~n')
    ),

    % Test 4: Declare custom interaction
    format('~nTest 4: Declare custom interaction...~n'),
    declare_mindmap_interaction(test_map, [on_node_click(custom_action)]),
    (   get_mindmap_event_handler(test_map, on_node_click, custom_action)
    ->  format('  PASS: Custom interaction declared~n')
    ;   format('  FAIL: Custom declaration failed~n')
    ),

    % Test 5: Edit mode interactions
    format('~nTest 5: Edit mode interactions...~n'),
    (   mindmap_interaction(edit_mode, EditEvents),
        member(on_background_double_click(create_node), EditEvents)
    ->  format('  PASS: Edit mode has create_node~n')
    ;   format('  FAIL: Edit mode missing create_node~n')
    ),

    % Cleanup
    retractall(mindmap_interaction(test_map, _)),

    format('~n=== Tests Complete ===~n').

%% test_gesture_support
%
%  Test touch gesture support functionality.
%
test_gesture_support :-
    format('~n=== Touch Gesture Support Tests ===~n~n'),

    % Test 1: Default gesture spec exists
    format('Test 1: Default gesture spec...~n'),
    (   mindmap_gesture_spec(default, GestureSpec),
        member(gesture(tap, _), GestureSpec)
    ->  format('  PASS: Default gesture spec defined~n')
    ;   format('  FAIL: Default gesture spec missing~n')
    ),

    % Test 2: Edit mode gesture spec
    format('~nTest 2: Edit mode gestures...~n'),
    (   mindmap_gesture_spec(edit_mode, EditSpec),
        member(gesture(long_press, LPOpts), EditSpec),
        member(action(start_drag), LPOpts)
    ->  format('  PASS: Edit mode has long_press for drag~n')
    ;   format('  FAIL: Edit mode long_press missing~n')
    ),

    % Test 3: Generate gesture handlers
    format('~nTest 3: Generate gesture handlers...~n'),
    generate_gesture_handlers(default, Code),
    (   sub_string(Code, _, _, _, "initGestures"),
        sub_string(Code, _, _, _, "Hammer")
    ->  format('  PASS: Gesture handlers generated~n')
    ;   format('  FAIL: Gesture handler generation failed~n')
    ),

    % Test 4: Gesture config generation
    format('~nTest 4: Gesture config...~n'),
    (   sub_string(Code, _, _, _, "gestureConfig"),
        sub_string(Code, _, _, _, "touchAction")
    ->  format('  PASS: Gesture config included~n')
    ;   format('  FAIL: Gesture config missing~n')
    ),

    % Test 5: Pinch gesture with scale limits
    format('~nTest 5: Pinch gesture scale limits...~n'),
    (   mindmap_gesture_spec(default, DefSpec),
        member(gesture(pinch, PinchOpts), DefSpec),
        member(min_scale(MinS), PinchOpts),
        member(max_scale(MaxS), PinchOpts),
        MinS < MaxS
    ->  format('  PASS: Pinch has scale limits (~w to ~w)~n', [MinS, MaxS])
    ;   format('  FAIL: Pinch scale limits incorrect~n')
    ),

    % Test 6: Swipe gestures for navigation
    format('~nTest 6: Swipe gestures...~n'),
    (   mindmap_gesture_spec(default, SwipeSpec),
        member(gesture(swipe_left, SwipeLeftOpts), SwipeSpec),
        member(gesture(swipe_right, SwipeRightOpts), SwipeSpec),
        member(action(next_sibling), SwipeLeftOpts),
        member(action(previous_sibling), SwipeRightOpts)
    ->  format('  PASS: Swipe navigation configured~n')
    ;   format('  FAIL: Swipe navigation missing~n')
    ),

    % Test 7: Presentation mode gestures
    format('~nTest 7: Presentation mode gestures...~n'),
    (   mindmap_gesture_spec(presentation, PresSpec),
        member(gesture(swipe_left, PresSwipeOpts), PresSpec),
        member(action(next_slide), PresSwipeOpts)
    ->  format('  PASS: Presentation gestures configured~n')
    ;   format('  FAIL: Presentation gestures missing~n')
    ),

    % Test 8: Declare custom gesture spec
    format('~nTest 8: Declare custom gesture spec...~n'),
    declare_mindmap_gesture_spec(custom_test, [
        gesture(tap, [fingers(1), action(custom_action)])
    ]),
    (   get_gesture_spec(custom_test, CustomSpec),
        member(gesture(tap, CustomTapOpts), CustomSpec),
        member(action(custom_action), CustomTapOpts)
    ->  format('  PASS: Custom gesture spec declared~n')
    ;   format('  FAIL: Custom gesture declaration failed~n')
    ),

    % Cleanup
    retractall(mindmap_gesture_spec(custom_test, _)),

    format('~n=== Gesture Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind map interaction module loaded~n', [])
), now).

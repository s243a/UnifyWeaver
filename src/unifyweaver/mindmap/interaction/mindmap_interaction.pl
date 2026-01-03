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

    % Generation predicates
    generate_mindmap_interaction/2,
    generate_mindmap_handlers/2,
    generate_mindmap_events_jsx/2,

    % Management
    declare_mindmap_interaction/2,
    clear_mindmap_interactions/0,

    % Queries
    has_mindmap_interaction/2,
    get_mindmap_event_handler/3,

    % Testing
    test_mindmap_interaction/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic mindmap_interaction/2.
:- dynamic mindmap_tooltip_spec/2.
:- dynamic mindmap_selection_spec/2.
:- dynamic mindmap_drag_spec/2.

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
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_mindmap_interaction(+MapId, +Events)
declare_mindmap_interaction(MapId, Events) :-
    retractall(mindmap_interaction(MapId, _)),
    assertz(mindmap_interaction(MapId, Events)).

%% clear_mindmap_interactions
clear_mindmap_interactions :-
    retractall(mindmap_interaction(_, _)),
    retractall(mindmap_tooltip_spec(_, _)),
    retractall(mindmap_selection_spec(_, _)),
    retractall(mindmap_drag_spec(_, _)).

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

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind map interaction module loaded~n', [])
), now).

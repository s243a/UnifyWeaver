%% test_react_native_codegen.pl - plunit tests for React Native code generation
%%
%% Tests React Native component code generation from react_native_target module.
%%
%% Run with: swipl -g "run_tests" -t halt test_react_native_codegen.pl

:- module(test_react_native_codegen, []).

:- use_module(library(plunit)).

%% Load React Native target module
:- use_module('../../targets/react_native_target').

%% ============================================================================
%% Test Data
%% ============================================================================

test_nodes([
    node(root, [label("Main Topic"), type(root)]),
    node(child1, [label("Child 1"), type(hub)]),
    node(child2, [label("Child 2"), type(branch)]),
    node(leaf1, [label("Leaf Node"), type(leaf)])
]).

test_edges([
    edge(root, child1, []),
    edge(root, child2, []),
    edge(child1, leaf1, [type(strong)])
]).

%% ============================================================================
%% Tests: React Native Target Capabilities
%% ============================================================================

:- begin_tests(rn_capabilities).

test(rn_supports_svg) :-
    react_native_target:rn_capabilities(Caps),
    member(supports(svg_graphics), Caps).

test(rn_supports_gestures) :-
    react_native_target:rn_capabilities(Caps),
    member(supports(gesture_handling), Caps).

test(rn_supports_animations) :-
    react_native_target:rn_capabilities(Caps),
    member(supports(animated_transitions), Caps).

test(rn_has_svg_library) :-
    react_native_target:rn_capabilities(Caps),
    member(library('react-native-svg'), Caps).

test(rn_has_gesture_library) :-
    react_native_target:rn_capabilities(Caps),
    member(library('react-native-gesture-handler'), Caps).

test(rn_has_reanimated_library) :-
    react_native_target:rn_capabilities(Caps),
    member(library('react-native-reanimated'), Caps).

test(rn_has_limitations_defined) :-
    react_native_target:rn_capabilities(Caps),
    member(limitation(no_dom_apis), Caps).

:- end_tests(rn_capabilities).

%% ============================================================================
%% Tests: Predicate Compilation
%% ============================================================================

:- begin_tests(rn_predicate_compilation).

test(rn_compiles_predicate_imports_react) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "import React from 'react'").

test(rn_compiles_predicate_imports_react_native) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "from 'react-native'").

test(rn_compiles_predicate_has_view) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "View").

test(rn_compiles_predicate_has_text) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "Text").

test(rn_compiles_predicate_has_stylesheet) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "StyleSheet.create").

test(rn_compiles_predicate_has_component_name) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "Test_pred").

test(rn_compiles_predicate_has_module_ref) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "test_module:test_pred/2").

test(rn_compiles_predicate_exports_default) :-
    react_native_target:compile_predicate_to_react_native(test_module:test_pred/2, [], Code),
    sub_string(Code, _, _, _, "export default").

:- end_tests(rn_predicate_compilation).

%% ============================================================================
%% Tests: Mindmap Component Generation
%% ============================================================================

:- begin_tests(rn_mindmap_component).

test(rn_mindmap_imports_svg) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "react-native-svg").

test(rn_mindmap_imports_gesture_handler) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "react-native-gesture-handler").

test(rn_mindmap_imports_reanimated) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "react-native-reanimated").

test(rn_mindmap_has_gesture_detector) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "GestureDetector").

test(rn_mindmap_has_animated_view) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Animated.View").

test(rn_mindmap_has_svg_elements) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Svg"),
    sub_string(Code, _, _, _, "Ellipse"),
    sub_string(Code, _, _, _, "Line").

test(rn_mindmap_has_pinch_gesture) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Gesture.Pinch").

test(rn_mindmap_has_pan_gesture) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Gesture.Pan").

test(rn_mindmap_embeds_node_data) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Main Topic"),
    sub_string(Code, _, _, _, "Child 1").

test(rn_mindmap_embeds_edge_data) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "source: 'root'"),
    sub_string(Code, _, _, _, "target: 'child1'").

:- end_tests(rn_mindmap_component).

%% ============================================================================
%% Tests: List Component Generation
%% ============================================================================

:- begin_tests(rn_list_component).

test(rn_list_has_flatlist) :-
    Items = [item(id1, "Item 1"), item(id2, "Item 2")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "FlatList").

test(rn_list_has_render_item) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "renderItem").

test(rn_list_has_key_extractor) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "keyExtractor").

test(rn_list_has_pressable) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "Pressable").

test(rn_list_embeds_items) :-
    Items = [item(id1, "Test Item")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "Test Item").

test(rn_list_has_typed_interface) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "interface Item").

test(rn_list_respects_component_name, [nondet]) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [component_name('MyList')], Code),
    sub_string(Code, _, _, _, "MyList").

:- end_tests(rn_list_component).

%% ============================================================================
%% Tests: Card Component Generation
%% ============================================================================

:- begin_tests(rn_card_component).

test(rn_card_has_pressable) :-
    react_native_target:generate_rn_card_component([title("Test")], [], Code),
    sub_string(Code, _, _, _, "Pressable").

test(rn_card_has_image_support) :-
    react_native_target:generate_rn_card_component([title("Test")], [], Code),
    sub_string(Code, _, _, _, "Image"),
    sub_string(Code, _, _, _, "imageUrl").

test(rn_card_has_shadow_styles) :-
    react_native_target:generate_rn_card_component([title("Test")], [], Code),
    sub_string(Code, _, _, _, "shadowColor"),
    sub_string(Code, _, _, _, "shadowOffset").

test(rn_card_has_elevation_android) :-
    react_native_target:generate_rn_card_component([title("Test")], [], Code),
    sub_string(Code, _, _, _, "elevation").

test(rn_card_embeds_title) :-
    react_native_target:generate_rn_card_component([title("Custom Title")], [], Code),
    sub_string(Code, _, _, _, "Custom Title").

test(rn_card_respects_component_name, [nondet]) :-
    react_native_target:generate_rn_card_component([title("Test")], [component_name('MyCard')], Code),
    sub_string(Code, _, _, _, "MyCard").

:- end_tests(rn_card_component).

%% ============================================================================
%% Tests: Theme Support
%% ============================================================================

:- begin_tests(rn_theme_support).

test(rn_light_theme_has_white_bg) :-
    react_native_target:generate_theme_colors_rn(light, Code),
    sub_string(Code, _, _, _, "#ffffff").

test(rn_dark_theme_has_dark_bg) :-
    react_native_target:generate_theme_colors_rn(dark, Code),
    sub_string(Code, _, _, _, "#1a1a2e").

test(rn_theme_has_node_colors) :-
    react_native_target:generate_theme_colors_rn(light, Code),
    sub_string(Code, _, _, _, "nodeColors"),
    sub_string(Code, _, _, _, "root"),
    sub_string(Code, _, _, _, "hub"),
    sub_string(Code, _, _, _, "leaf").

test(rn_theme_has_edge_color) :-
    react_native_target:generate_theme_colors_rn(light, Code),
    sub_string(Code, _, _, _, "edgeColor").

test(rn_mindmap_respects_theme, [nondet]) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [theme(dark)], Code),
    sub_string(Code, _, _, _, "#1a1a2e").

:- end_tests(rn_theme_support).

%% ============================================================================
%% Tests: TypeScript Support
%% ============================================================================

:- begin_tests(rn_typescript).

test(rn_mindmap_has_interfaces) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "interface MindMapNode"),
    sub_string(Code, _, _, _, "interface MindMapEdge").

test(rn_mindmap_has_typed_props) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "React.FC<").

test(rn_mindmap_has_typed_state) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "useSharedValue").

test(rn_mindmap_has_generic_record) :-
    test_nodes(Nodes),
    test_edges(Edges),
    react_native_target:generate_rn_mindmap_component(Nodes, Edges, [], Code),
    sub_string(Code, _, _, _, "Record<string, MindMapNode>").

test(rn_list_has_typed_item) :-
    Items = [item(id1, "Item 1")],
    react_native_target:generate_rn_list_component(Items, [], Code),
    sub_string(Code, _, _, _, "{ item }: { item: Item }").

test(rn_card_has_typed_props_interface) :-
    react_native_target:generate_rn_card_component([title("Test")], [], Code),
    sub_string(Code, _, _, _, "interface").

:- end_tests(rn_typescript).

%% ============================================================================
%% Tests: Platform-Specific Code
%% ============================================================================

:- begin_tests(rn_platform_specific).

test(rn_platform_ios_returns_ios_code) :-
    react_native_target:rn_platform_specific(ios, "ios_code", "android_code", Result),
    Result == "ios_code".

test(rn_platform_android_returns_android_code) :-
    react_native_target:rn_platform_specific(android, "ios_code", "android_code", Result),
    Result == "android_code".

test(rn_platform_both_generates_select) :-
    react_native_target:rn_platform_specific(both, "ios_val", "android_val", Result),
    sub_string(Result, _, _, _, "Platform.select"),
    sub_string(Result, _, _, _, "ios"),
    sub_string(Result, _, _, _, "android").

:- end_tests(rn_platform_specific).

%% ============================================================================
%% Tests: Style Generation
%% ============================================================================

:- begin_tests(rn_style_generation).

test(rn_styles_generates_stylesheet) :-
    Styles = [style(container, [flex(1), padding(16)])],
    react_native_target:generate_rn_styles(Styles, Code),
    sub_string(Code, _, _, _, "StyleSheet.create").

test(rn_styles_includes_style_name) :-
    Styles = [style(container, [flex(1)])],
    react_native_target:generate_rn_styles(Styles, Code),
    sub_string(Code, _, _, _, "container").

test(rn_styles_includes_numeric_values) :-
    Styles = [style(box, [width(100), height(50)])],
    react_native_target:generate_rn_styles(Styles, Code),
    sub_string(Code, _, _, _, "width: 100"),
    sub_string(Code, _, _, _, "height: 50").

:- end_tests(rn_style_generation).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization(run_tests, main).

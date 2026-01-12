% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_native_target.pl - React Native Code Generation Target
%
% Generates React Native components from Prolog predicates for mobile apps.
% Supports:
% - react-native-svg for vector graphics
% - React Native Animated API for animations
% - Touch handling (Pressable, gesture handlers)
% - Expo and bare React Native projects
%
% Usage:
%   ?- compile_predicate_to_react_native(my_module:my_pred/2, [], RNCode).

:- module(react_native_target, [
    compile_predicate_to_react_native/3,
    compile_component_to_react_native/4,
    generate_rn_mindmap_component/4,
    generate_rn_list_component/3,
    generate_rn_card_component/3,
    init_react_native_target/0,
    test_react_native_target/0,
    % Platform detection
    rn_platform_specific/4,
    rn_capabilities/1,
    % Style generation
    generate_rn_styles/2
]).

:- use_module(library(lists)).

% ============================================================================
% INITIALIZATION
% ============================================================================

init_react_native_target :-
    format('React Native target initialized~n', []).

% ============================================================================
% CAPABILITIES
% ============================================================================

%% rn_capabilities(-Capabilities)
%
%  Lists React Native target capabilities.
%
rn_capabilities([
    % Supported features
    supports(svg_graphics),
    supports(animated_transitions),
    supports(gesture_handling),
    supports(platform_specific_code),
    supports(expo_compatibility),
    supports(typescript),
    % Libraries used
    library('react-native-svg'),
    library('react-native-gesture-handler'),
    library('react-native-reanimated'),
    % Limitations
    limitation(no_dom_apis),
    limitation(no_css_animations),
    limitation(limited_svg_filters)
]).

%% rn_platform_specific(+Platform, +IOSCode, +AndroidCode)
%
%  Generate platform-specific code.
%
rn_platform_specific(ios, IOSCode, _, IOSCode).
rn_platform_specific(android, _, AndroidCode, AndroidCode).
rn_platform_specific(both, IOSCode, AndroidCode, Code) :-
    format(string(Code),
"Platform.select({
  ios: () => ~w,
  android: () => ~w,
})()
", [IOSCode, AndroidCode]).

% ============================================================================
% PREDICATE COMPILATION
% ============================================================================

%% compile_predicate_to_react_native(:Pred, +Options, -RNCode)
%
%  Compile a Prolog predicate to a React Native component.
%
compile_predicate_to_react_native(Module:Pred/Arity, Options, RNCode) :-
    atom_string(Pred, PredStr),
    atom_string(Module, ModStr),
    option_value(Options, component_type, functional, _ComponentType),
    option_value(Options, typescript, true, UseTS),

    capitalize_first(PredStr, ComponentName),
    generate_arg_props(Arity, PropsInterface),

    (   UseTS == true
    ->  TypeAnnotations = true
    ;   TypeAnnotations = false
    ),

    generate_rn_component(ComponentName, PropsInterface, TypeAnnotations, ModStr, PredStr, Arity, RNCode).

generate_rn_component(Name, PropsInterface, TypeAnnotations, Module, Pred, Arity, RNCode) :-
    (   TypeAnnotations == true
    ->  PropsType = PropsInterface,
        format(string(PropsDecl), "interface ~wProps {\n~w\n}\n\n", [Name, PropsType])
    ;   PropsDecl = ""
    ),
    
    format(string(RNCode),
"/**
 * ~w - React Native Component
 * Generated from Prolog predicate: ~w:~w/~w
 */

import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';

~wexport const ~w: React.FC<~wProps> = (props) => {
  // TODO: Implement predicate logic
  
  return (
    <View style={styles.container}>
      <Text style={styles.title}>~w</Text>
      <Text style={styles.subtitle}>From: ~w:~w/~w</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1a202c',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#718096',
  },
});

export default ~w;
", [Name, Module, Pred, Arity, PropsDecl, Name, Name, Name, Module, Pred, Arity, Name]).

generate_arg_props(0, "  // No props") :- !.
generate_arg_props(Arity, Props) :-
    findall(PropLine, (
        between(1, Arity, N),
        format(string(PropLine), "  arg~w?: any;", [N])
    ), Lines),
    atomic_list_concat(Lines, '\n', Props).

% ============================================================================
% COMPONENT TEMPLATES
% ============================================================================

%% compile_component_to_react_native(+Name, +Props, +Children, -RNCode)
compile_component_to_react_native(Name, Props, Children, RNCode) :-
    generate_props_interface(Props, PropsInterface),
    generate_children_render(Children, ChildrenRender),
    format(string(RNCode),
"import React from 'react';
import { View, Text, StyleSheet, FlatList, Pressable } from 'react-native';

~w

export const ~w: React.FC<~wProps> = ({ children, ...props }) => {
  return (
    <View style={styles.container}>
~w
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default ~w;
", [PropsInterface, Name, Name, ChildrenRender, Name]).

generate_props_interface(Props, Interface) :-
    findall(PropLine, (
        member(prop(PropName, PropType), Props),
        format(string(PropLine), "  ~w: ~w;", [PropName, PropType])
    ), Lines),
    (   Lines = []
    ->  PropsBody = "  children?: React.ReactNode;"
    ;   atomic_list_concat(Lines, '\n', PropsLines),
        format(string(PropsBody), "~w\n  children?: React.ReactNode;", [PropsLines])
    ),
    format(string(Interface), "interface Props {\n~w\n}", [PropsBody]).

generate_children_render([], "      {children}").
generate_children_render(Children, Render) :-
    Children \= [],
    maplist(child_to_jsx, Children, JSXChildren),
    atomic_list_concat(JSXChildren, '\n', Render).

child_to_jsx(text(Content), JSX) :-
    format(string(JSX), "      <Text>~w</Text>", [Content]).
child_to_jsx(view(Children), JSX) :-
    generate_children_render(Children, ChildRender),
    format(string(JSX), "      <View>\n~w\n      </View>", [ChildRender]).

% ============================================================================
% MINDMAP COMPONENT
% ============================================================================

%% generate_rn_mindmap_component(+Nodes, +Edges, +Options, -RNCode)
%
%  Generate a React Native mindmap visualization component using react-native-svg.
%
generate_rn_mindmap_component(Nodes, Edges, Options, RNCode) :-
    option_value(Options, component_name, 'MindMap', ComponentName),
    option_value(Options, width, 400, Width),
    option_value(Options, height, 300, Height),
    option_value(Options, theme, light, Theme),
    
    generate_nodes_data(Nodes, NodesData),
    generate_edges_data(Edges, EdgesData),
    generate_theme_colors_rn(Theme, ThemeColors),
    
    format(string(RNCode),
"/**
 * ~w - React Native Mind Map Component
 * Generated by UnifyWeaver
 * 
 * Uses react-native-svg for vector graphics rendering.
 * Supports pan and zoom via gesture handlers.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Svg, { G, Circle, Ellipse, Line, Text as SvgText } from 'react-native-svg';
import { GestureDetector, Gesture, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';

// Types
interface MindMapNode {
  id: string;
  label: string;
  type: 'root' | 'hub' | 'branch' | 'leaf' | 'default';
  x?: number;
  y?: number;
  url?: string;
}

interface MindMapEdge {
  source: string;
  target: string;
  type?: string;
}

interface ~wProps {
  nodes?: MindMapNode[];
  edges?: MindMapEdge[];
  width?: number;
  height?: number;
  onNodePress?: (node: MindMapNode) => void;
}

// Theme colors
~w

// Default data
const defaultNodes: MindMapNode[] = ~w;
const defaultEdges: MindMapEdge[] = ~w;

// Simple force layout calculation
const calculateLayout = (nodes: MindMapNode[], width: number, height: number) => {
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 3;
  
  return nodes.map((node, index) => {
    if (node.x !== undefined && node.y !== undefined) {
      return node;
    }
    
    if (node.type === 'root') {
      return { ...node, x: centerX, y: centerY };
    }
    
    const angle = (2 * Math.PI * index) / nodes.length;
    const distance = node.type === 'hub' ? radius * 0.6 : radius;
    
    return {
      ...node,
      x: centerX + Math.cos(angle) * distance,
      y: centerY + Math.sin(angle) * distance,
    };
  });
};

export const ~w: React.FC<~wProps> = ({
  nodes = defaultNodes,
  edges = defaultEdges,
  width = ~w,
  height = ~w,
  onNodePress,
}) => {
  // Gesture state
  const scale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedScale = useSharedValue(1);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  // Calculate node positions
  const layoutNodes = useMemo(
    () => calculateLayout(nodes, width, height),
    [nodes, width, height]
  );

  // Create node position map for edge rendering
  const nodeMap = useMemo(() => {
    const map: Record<string, MindMapNode> = {};
    layoutNodes.forEach(node => {
      map[node.id] = node;
    });
    return map;
  }, [layoutNodes]);

  // Gestures
  const pinchGesture = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = savedScale.value * e.scale;
    })
    .onEnd(() => {
      savedScale.value = scale.value;
    });

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      translateX.value = savedTranslateX.value + e.translationX;
      translateY.value = savedTranslateY.value + e.translationY;
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const composedGesture = Gesture.Simultaneous(pinchGesture, panGesture);

  // Animated style
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  // Node press handler
  const handleNodePress = useCallback((node: MindMapNode) => {
    if (onNodePress) {
      onNodePress(node);
    }
  }, [onNodePress]);

  // Get node style based on type
  const getNodeStyle = (type: string) => {
    return themeColors.nodeColors[type] || themeColors.nodeColors.default;
  };

  return (
    <GestureHandlerRootView style={styles.container}>
      <GestureDetector gesture={composedGesture}>
        <Animated.View style={[styles.svgContainer, animatedStyle]}>
          <Svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
            <G>
              {/* Render edges */}
              {edges.map((edge, index) => {
                const source = nodeMap[edge.source];
                const target = nodeMap[edge.target];
                if (!source || !target) return null;
                
                return (
                  <Line
                    key={`edge-${index}`}
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    stroke={themeColors.edgeColor}
                    strokeWidth={2}
                  />
                );
              })}
              
              {/* Render nodes */}
              {layoutNodes.map((node) => {
                const style = getNodeStyle(node.type);
                const rx = node.type === 'root' ? 50 : 40;
                const ry = node.type === 'root' ? 30 : 25;
                
                return (
                  <G
                    key={node.id}
                    onPress={() => handleNodePress(node)}
                  >
                    <Ellipse
                      cx={node.x}
                      cy={node.y}
                      rx={rx}
                      ry={ry}
                      fill={style.fill}
                      stroke={style.stroke}
                      strokeWidth={node.type === 'root' ? 3 : 2}
                    />
                    <SvgText
                      x={node.x}
                      y={node.y}
                      fill={style.text}
                      fontSize={node.type === 'root' ? 14 : 12}
                      fontWeight={node.type === 'root' ? 'bold' : 'normal'}
                      textAnchor=\"middle\"
                      alignmentBaseline=\"middle\"
                    >
                      {node.label.length > 12 ? node.label.slice(0, 10) + '...' : node.label}
                    </SvgText>
                  </G>
                );
              })}
            </G>
          </Svg>
        </Animated.View>
      </GestureDetector>
    </GestureHandlerRootView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  svgContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default ~w;
", [ComponentName, ComponentName, ThemeColors, NodesData, EdgesData, ComponentName, ComponentName,
    Width, Height, ComponentName]).

generate_nodes_data([], "[]").
generate_nodes_data(Nodes, Data) :-
    maplist(node_to_rn_json, Nodes, NodeStrs),
    atomic_list_concat(NodeStrs, ',\n  ', NodesConcat),
    format(string(Data), "[\n  ~w\n]", [NodesConcat]).

node_to_rn_json(node(Id, Props), JSON) :-
    atom_string(Id, IdStr),
    (   member(label(Label), Props)
    ->  true
    ;   Label = IdStr
    ),
    (   member(type(Type), Props)
    ->  atom_string(Type, TypeStr)
    ;   TypeStr = "default"
    ),
    format(string(JSON), "{ id: '~w', label: '~w', type: '~w' as const }", [IdStr, Label, TypeStr]).

generate_edges_data([], "[]").
generate_edges_data(Edges, Data) :-
    maplist(edge_to_rn_json, Edges, EdgeStrs),
    atomic_list_concat(EdgeStrs, ',\n  ', EdgesConcat),
    format(string(Data), "[\n  ~w\n]", [EdgesConcat]).

edge_to_rn_json(edge(From, To, _Props), JSON) :-
    atom_string(From, FromStr),
    atom_string(To, ToStr),
    format(string(JSON), "{ source: '~w', target: '~w' }", [FromStr, ToStr]).

generate_theme_colors_rn(light, Code) :-
    Code = "const themeColors = {
  background: '#ffffff',
  nodeColors: {
    default: { fill: '#e8f4fc', stroke: '#4a90d9', text: '#333333' },
    root: { fill: '#4a90d9', stroke: '#2c5a8c', text: '#ffffff' },
    hub: { fill: '#6ab04c', stroke: '#4a904c', text: '#ffffff' },
    branch: { fill: '#f0932b', stroke: '#c07020', text: '#ffffff' },
    leaf: { fill: '#eb4d4b', stroke: '#cb2d2b', text: '#ffffff' },
  },
  edgeColor: '#666666',
};".

generate_theme_colors_rn(dark, Code) :-
    Code = "const themeColors = {
  background: '#1a1a2e',
  nodeColors: {
    default: { fill: '#2d3748', stroke: '#4a9ce9', text: '#e2e8f0' },
    root: { fill: '#5a9ce9', stroke: '#3c6a9c', text: '#ffffff' },
    hub: { fill: '#7ac05c', stroke: '#5aa05c', text: '#ffffff' },
    branch: { fill: '#ffaa4b', stroke: '#d08030', text: '#000000' },
    leaf: { fill: '#fb5d5b', stroke: '#db3d3b', text: '#ffffff' },
  },
  edgeColor: '#718096',
};".

generate_theme_colors_rn(_, Code) :-
    generate_theme_colors_rn(light, Code).

% ============================================================================
% LIST COMPONENT
% ============================================================================

%% generate_rn_list_component(+Items, +Options, -RNCode)
%
%  Generate a React Native FlatList component.
%
generate_rn_list_component(Items, Options, RNCode) :-
    option_value(Options, component_name, 'ItemList', ComponentName),
    option_value(Options, item_component, 'default', _ItemComponent),

    generate_items_data(Items, ItemsData),
    
    format(string(RNCode),
"import React from 'react';
import { View, Text, FlatList, StyleSheet, Pressable } from 'react-native';

interface Item {
  id: string;
  title: string;
  subtitle?: string;
}

interface ~wProps {
  items?: Item[];
  onItemPress?: (item: Item) => void;
}

const defaultItems: Item[] = ~w;

export const ~w: React.FC<~wProps> = ({
  items = defaultItems,
  onItemPress,
}) => {
  const renderItem = ({ item }: { item: Item }) => (
    <Pressable
      style={styles.item}
      onPress={() => onItemPress?.(item)}
    >
      <Text style={styles.title}>{item.title}</Text>
      {item.subtitle && <Text style={styles.subtitle}>{item.subtitle}</Text>}
    </Pressable>
  );

  return (
    <FlatList
      data={items}
      renderItem={renderItem}
      keyExtractor={(item) => item.id}
      contentContainerStyle={styles.container}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  item: {
    backgroundColor: '#fff',
    padding: 16,
    marginBottom: 12,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1a202c',
  },
  subtitle: {
    fontSize: 14,
    color: '#718096',
    marginTop: 4,
  },
});

export default ~w;
", [ComponentName, ItemsData, ComponentName, ComponentName, ComponentName]).

generate_items_data([], "[]").
generate_items_data(Items, Data) :-
    maplist(item_to_json, Items, ItemStrs),
    atomic_list_concat(ItemStrs, ', ', ItemsConcat),
    format(string(Data), "[~w]", [ItemsConcat]).

item_to_json(item(Id, Title), JSON) :-
    format(string(JSON), "{ id: '~w', title: '~w' }", [Id, Title]).
item_to_json(item(Id, Title, Subtitle), JSON) :-
    format(string(JSON), "{ id: '~w', title: '~w', subtitle: '~w' }", [Id, Title, Subtitle]).

% ============================================================================
% CARD COMPONENT
% ============================================================================

%% generate_rn_card_component(+Data, +Options, -RNCode)
%
%  Generate a React Native card component.
%
generate_rn_card_component(Data, Options, RNCode) :-
    option_value(Options, component_name, 'Card', ComponentName),
    
    (   member(title(Title), Data) -> true ; Title = "Card Title" ),
    (   member(content(Content), Data) -> true ; Content = "Card content goes here." ),
    
    format(string(RNCode),
"import React from 'react';
import { View, Text, StyleSheet, Pressable, Image } from 'react-native';

interface ~wProps {
  title?: string;
  content?: string;
  imageUrl?: string;
  onPress?: () => void;
}

export const ~w: React.FC<~wProps> = ({
  title = '~w',
  content = '~w',
  imageUrl,
  onPress,
}) => {
  return (
    <Pressable style={styles.card} onPress={onPress}>
      {imageUrl && (
        <Image source={{ uri: imageUrl }} style={styles.image} />
      )}
      <View style={styles.content}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.text}>{content}</Text>
      </View>
    </Pressable>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: 160,
    resizeMode: 'cover',
  },
  content: {
    padding: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1a202c',
    marginBottom: 8,
  },
  text: {
    fontSize: 14,
    color: '#4a5568',
    lineHeight: 20,
  },
});

export default ~w;
", [ComponentName, ComponentName, ComponentName, Title, Content, ComponentName]).

% ============================================================================
% STYLE GENERATION
% ============================================================================

%% generate_rn_styles(+StyleSpec, -StyleSheet)
generate_rn_styles(Styles, StyleSheet) :-
    maplist(style_to_rn, Styles, StyleStrs),
    atomic_list_concat(StyleStrs, ',\n  ', StylesConcat),
    format(string(StyleSheet), "const styles = StyleSheet.create({\n  ~w\n});", [StylesConcat]).

style_to_rn(style(Name, Props), StyleStr) :-
    maplist(prop_to_rn, Props, PropStrs),
    atomic_list_concat(PropStrs, ', ', PropsConcat),
    format(string(StyleStr), "~w: { ~w }", [Name, PropsConcat]).

prop_to_rn(Prop, PropStr) :-
    Prop =.. [Key, Value],
    (   number(Value)
    ->  format(string(PropStr), "~w: ~w", [Key, Value])
    ;   format(string(PropStr), "~w: '~w'", [Key, Value])
    ).

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HUC]),
    string_chars(Cap, [HUC|T]).

% ============================================================================
% TESTING
% ============================================================================

test_react_native_target :-
    format('~n=== React Native Target Tests ===~n~n'),

    % Test 1: Capabilities
    format('Test 1: Capabilities check...~n'),
    rn_capabilities(Caps),
    (   member(supports(svg_graphics), Caps),
        member(library('react-native-svg'), Caps)
    ->  format('  PASS: Capabilities defined~n')
    ;   format('  FAIL: Capabilities incorrect~n')
    ),

    % Test 2: Predicate compilation
    format('~nTest 2: Predicate compilation...~n'),
    compile_predicate_to_react_native(test_module:test_pred/2, [], Code1),
    (   sub_string(Code1, _, _, _, "import React"),
        sub_string(Code1, _, _, _, "react-native"),
        sub_string(Code1, _, _, _, "StyleSheet")
    ->  format('  PASS: Predicate compiled~n')
    ;   format('  FAIL: Predicate compilation failed~n')
    ),

    % Test 3: Mindmap component
    format('~nTest 3: Mindmap component generation...~n'),
    TestNodes = [node(root, [label("Root"), type(root)]), node(child, [label("Child"), type(leaf)])],
    TestEdges = [edge(root, child, [])],
    generate_rn_mindmap_component(TestNodes, TestEdges, [], Code2),
    (   sub_string(Code2, _, _, _, "react-native-svg"),
        sub_string(Code2, _, _, _, "GestureDetector"),
        sub_string(Code2, _, _, _, "Animated")
    ->  format('  PASS: Mindmap component generated~n')
    ;   format('  FAIL: Mindmap component failed~n')
    ),

    % Test 4: List component
    format('~nTest 4: List component generation...~n'),
    TestItems = [item(id1, "Item 1"), item(id2, "Item 2", "Subtitle")],
    generate_rn_list_component(TestItems, [], Code3),
    (   sub_string(Code3, _, _, _, "FlatList"),
        sub_string(Code3, _, _, _, "renderItem")
    ->  format('  PASS: List component generated~n')
    ;   format('  FAIL: List component failed~n')
    ),

    % Test 5: Card component
    format('~nTest 5: Card component generation...~n'),
    generate_rn_card_component([title("Test Card")], [], Code4),
    (   sub_string(Code4, _, _, _, "Pressable"),
        sub_string(Code4, _, _, _, "Test Card")
    ->  format('  PASS: Card component generated~n')
    ;   format('  FAIL: Card component failed~n')
    ),

    % Test 6: Theme colors
    format('~nTest 6: Theme colors...~n'),
    generate_theme_colors_rn(light, LightTheme),
    generate_theme_colors_rn(dark, DarkTheme),
    (   sub_string(LightTheme, _, _, _, "#ffffff"),
        sub_string(DarkTheme, _, _, _, "#1a1a2e")
    ->  format('  PASS: Theme colors generated~n')
    ;   format('  FAIL: Theme colors failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('React Native target module loaded~n', [])
), now).

% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_serializer.pl - JSON Lines Serialization for Mind Map DSL Objects
%
% Provides serialization/deserialization of mindmap objects (nodes, edges,
% positions, styles) to/from JSON Lines format for cross-target communication.
%
% This module follows the pipe_glue.pl patterns and integrates with the
% cross-target glue system, enabling mindmap data to flow between Prolog,
% Python, and other targets.
%
% Usage:
%   ?- serialize_mindmap_nodes(Nodes, JsonL).
%   ?- deserialize_mindmap_nodes(JsonL, Nodes).
%   ?- generate_mindmap_reader(python, Code).

:- module(mindmap_serializer, [
    % Object serialization (Prolog → JSON)
    serialize_node/2,               % serialize_node(+Node, -JsonDict)
    serialize_edge/2,               % serialize_edge(+Edge, -JsonDict)
    serialize_position/2,           % serialize_position(+Position, -JsonDict)
    serialize_style/2,              % serialize_style(+Style, -JsonDict)

    % Batch serialization
    serialize_mindmap_nodes/2,      % serialize_mindmap_nodes(+Nodes, -JsonL)
    serialize_mindmap_edges/2,      % serialize_mindmap_edges(+Edges, -JsonL)
    serialize_mindmap_positions/2,  % serialize_mindmap_positions(+Positions, -JsonL)
    serialize_mindmap_graph/2,      % serialize_mindmap_graph(+Graph, -JsonL)

    % Object deserialization (JSON → Prolog)
    deserialize_node/2,             % deserialize_node(+JsonDict, -Node)
    deserialize_edge/2,             % deserialize_edge(+JsonDict, -Edge)
    deserialize_position/2,         % deserialize_position(+JsonDict, -Position)

    % Batch deserialization
    deserialize_mindmap_nodes/2,    % deserialize_mindmap_nodes(+JsonL, -Nodes)
    deserialize_mindmap_edges/2,    % deserialize_mindmap_edges(+JsonL, -Edges)
    deserialize_mindmap_graph/2,    % deserialize_mindmap_graph(+JsonL, -Graph)

    % Stream I/O
    write_mindmap_jsonl/2,          % write_mindmap_jsonl(+Stream, +Objects)
    read_mindmap_jsonl/2,           % read_mindmap_jsonl(+Stream, -Objects)

    % Code generation for other targets
    generate_mindmap_reader/2,      % generate_mindmap_reader(+Target, -Code)
    generate_mindmap_writer/2,      % generate_mindmap_writer(+Target, -Code)

    % Protocol abstraction
    mindmap_protocol/2,             % mindmap_protocol(+ProtocolName, -Spec)
    register_mindmap_protocol/2,    % register_mindmap_protocol(+Name, +Spec)

    % Testing
    test_mindmap_serializer/0
]).

:- use_module(library(lists)).

% ============================================================================
% JSON LINES FORMAT SPECIFICATION
% ============================================================================
%
% Each mindmap object is serialized as a single JSON line with a "type" field:
%
% Node:
%   {"type":"node","id":"n1","label":"Topic","node_type":"root","props":{...}}
%
% Edge:
%   {"type":"edge","from":"n1","to":"n2","edge_type":"parent","props":{...}}
%
% Position:
%   {"type":"position","id":"n1","x":100.5,"y":200.3}
%
% Style:
%   {"type":"style","selector":"root","properties":{"fill":"#4a90d9",...}}
%
% Graph (compound object):
%   {"type":"graph","id":"map1","nodes":[...],"edges":[...],"positions":[...]}
%
% ============================================================================

% ============================================================================
% PROTOCOL REGISTRY
% ============================================================================

:- dynamic mindmap_protocol/2.

% Default protocol: JSON Lines
mindmap_protocol(jsonl, protocol{
    name: jsonl,
    description: "JSON Lines - one JSON object per line",
    mime_type: "application/x-ndjson",
    file_extension: ".jsonl",
    record_delimiter: "\n",
    supports_streaming: true,
    supports_nested: true
}).

% TSV protocol (flat records only)
mindmap_protocol(tsv, protocol{
    name: tsv,
    description: "Tab-separated values",
    mime_type: "text/tab-separated-values",
    file_extension: ".tsv",
    record_delimiter: "\n",
    field_delimiter: "\t",
    supports_streaming: true,
    supports_nested: false
}).

%% register_mindmap_protocol(+Name, +Spec)
register_mindmap_protocol(Name, Spec) :-
    retractall(mindmap_protocol(Name, _)),
    assertz(mindmap_protocol(Name, Spec)).

% ============================================================================
% NODE SERIALIZATION
% ============================================================================

%% serialize_node(+Node, -JsonDict)
%  Convert a mindmap node to a JSON-compatible dict.
%
serialize_node(node(Id, Props), JsonDict) :-
    % Extract standard properties
    (member(label(Label), Props) -> true ; Label = ""),
    (member(type(NodeType), Props) -> true ; NodeType = default),
    (member(parent(Parent), Props) -> true ; Parent = null),
    (member(link(Link), Props) -> true ; Link = null),
    (member(cluster(Cluster), Props) -> true ; Cluster = null),

    % Collect remaining properties
    exclude(is_standard_node_prop, Props, ExtraProps),
    props_to_dict(ExtraProps, ExtraDict),

    % Build JSON dict
    JsonDict = json{
        type: node,
        id: Id,
        label: Label,
        node_type: NodeType,
        parent: Parent,
        link: Link,
        cluster: Cluster,
        props: ExtraDict
    }.

is_standard_node_prop(label(_)).
is_standard_node_prop(type(_)).
is_standard_node_prop(parent(_)).
is_standard_node_prop(link(_)).
is_standard_node_prop(cluster(_)).

%% deserialize_node(+JsonDict, -Node)
deserialize_node(JsonDict, node(Id, Props)) :-
    Id = JsonDict.id,
    Label = JsonDict.label,
    NodeType = JsonDict.node_type,
    Parent = JsonDict.parent,
    Link = JsonDict.link,
    Cluster = JsonDict.cluster,

    % Build properties list
    Props0 = [label(Label), type(NodeType)],
    (Parent \== null -> Props1 = [parent(Parent) | Props0] ; Props1 = Props0),
    (Link \== null -> Props2 = [link(Link) | Props1] ; Props2 = Props1),
    (Cluster \== null -> Props3 = [cluster(Cluster) | Props2] ; Props3 = Props2),

    % Add extra props
    (   get_dict(props, JsonDict, ExtraDict),
        is_dict(ExtraDict)
    ->  dict_to_props(ExtraDict, ExtraProps),
        append(Props3, ExtraProps, Props)
    ;   Props = Props3
    ).

% ============================================================================
% EDGE SERIALIZATION
% ============================================================================

%% serialize_edge(+Edge, -JsonDict)
serialize_edge(edge(From, To, Props), JsonDict) :-
    (member(type(EdgeType), Props) -> true ; EdgeType = default),
    (member(weight(Weight), Props) -> true ; Weight = 1),
    (member(label(Label), Props) -> true ; Label = null),

    exclude(is_standard_edge_prop, Props, ExtraProps),
    props_to_dict(ExtraProps, ExtraDict),

    JsonDict = json{
        type: edge,
        from: From,
        to: To,
        edge_type: EdgeType,
        weight: Weight,
        label: Label,
        props: ExtraDict
    }.

is_standard_edge_prop(type(_)).
is_standard_edge_prop(weight(_)).
is_standard_edge_prop(label(_)).

%% deserialize_edge(+JsonDict, -Edge)
deserialize_edge(JsonDict, edge(From, To, Props)) :-
    From = JsonDict.from,
    To = JsonDict.to,
    EdgeType = JsonDict.edge_type,
    Weight = JsonDict.weight,
    Label = JsonDict.label,

    Props0 = [type(EdgeType), weight(Weight)],
    (Label \== null -> Props1 = [label(Label) | Props0] ; Props1 = Props0),

    (   get_dict(props, JsonDict, ExtraDict),
        is_dict(ExtraDict)
    ->  dict_to_props(ExtraDict, ExtraProps),
        append(Props1, ExtraProps, Props)
    ;   Props = Props1
    ).

% ============================================================================
% POSITION SERIALIZATION
% ============================================================================

%% serialize_position(+Position, -JsonDict)
serialize_position(position(Id, X, Y), JsonDict) :-
    JsonDict = json{
        type: position,
        id: Id,
        x: X,
        y: Y
    }.

%% deserialize_position(+JsonDict, -Position)
deserialize_position(JsonDict, position(Id, X, Y)) :-
    Id = JsonDict.id,
    X = JsonDict.x,
    Y = JsonDict.y.

% ============================================================================
% STYLE SERIALIZATION
% ============================================================================

%% serialize_style(+Style, -JsonDict)
serialize_style(style(Selector, Properties), JsonDict) :-
    props_to_dict(Properties, PropsDict),
    JsonDict = json{
        type: style,
        selector: Selector,
        properties: PropsDict
    }.

% ============================================================================
% BATCH SERIALIZATION
% ============================================================================

%% serialize_mindmap_nodes(+Nodes, -JsonL)
%  Serialize a list of nodes to JSON Lines string.
%
serialize_mindmap_nodes(Nodes, JsonL) :-
    maplist(serialize_node, Nodes, JsonDicts),
    maplist(dict_to_json_line, JsonDicts, Lines),
    atomic_list_concat(Lines, '\n', JsonL).

%% serialize_mindmap_edges(+Edges, -JsonL)
serialize_mindmap_edges(Edges, JsonL) :-
    maplist(serialize_edge, Edges, JsonDicts),
    maplist(dict_to_json_line, JsonDicts, Lines),
    atomic_list_concat(Lines, '\n', JsonL).

%% serialize_mindmap_positions(+Positions, -JsonL)
serialize_mindmap_positions(Positions, JsonL) :-
    maplist(serialize_position, Positions, JsonDicts),
    maplist(dict_to_json_line, JsonDicts, Lines),
    atomic_list_concat(Lines, '\n', JsonL).

%% serialize_mindmap_graph(+Graph, -JsonL)
%  Serialize a complete graph structure.
%  Graph = graph(Id, Nodes, Edges, Positions)
%
serialize_mindmap_graph(graph(Id, Nodes, Edges, Positions), JsonL) :-
    maplist(serialize_node, Nodes, NodeDicts),
    maplist(serialize_edge, Edges, EdgeDicts),
    maplist(serialize_position, Positions, PosDicts),

    GraphDict = json{
        type: graph,
        id: Id,
        nodes: NodeDicts,
        edges: EdgeDicts,
        positions: PosDicts
    },

    dict_to_json_line(GraphDict, JsonL).

% ============================================================================
% BATCH DESERIALIZATION
% ============================================================================

%% deserialize_mindmap_nodes(+JsonL, -Nodes)
deserialize_mindmap_nodes(JsonL, Nodes) :-
    split_json_lines(JsonL, Lines),
    maplist(json_line_to_dict, Lines, Dicts),
    include(is_node_dict, Dicts, NodeDicts),
    maplist(deserialize_node, NodeDicts, Nodes).

%% deserialize_mindmap_edges(+JsonL, -Edges)
deserialize_mindmap_edges(JsonL, Edges) :-
    split_json_lines(JsonL, Lines),
    maplist(json_line_to_dict, Lines, Dicts),
    include(is_edge_dict, Dicts, EdgeDicts),
    maplist(deserialize_edge, EdgeDicts, Edges).

%% deserialize_mindmap_graph(+JsonL, -Graph)
deserialize_mindmap_graph(JsonL, graph(Id, Nodes, Edges, Positions)) :-
    json_line_to_dict(JsonL, Dict),
    Dict.type == graph,
    Id = Dict.id,
    maplist(deserialize_node, Dict.nodes, Nodes),
    maplist(deserialize_edge, Dict.edges, Edges),
    maplist(deserialize_position, Dict.positions, Positions).

is_node_dict(Dict) :- Dict.type == node.
is_edge_dict(Dict) :- Dict.type == edge.

% ============================================================================
% STREAM I/O
% ============================================================================

%% write_mindmap_jsonl(+Stream, +Objects)
%  Write mindmap objects to a stream as JSON Lines.
%
write_mindmap_jsonl(Stream, Objects) :-
    forall(
        member(Obj, Objects),
        (   serialize_object(Obj, Dict),
            dict_to_json_line(Dict, Line),
            writeln(Stream, Line)
        )
    ).

serialize_object(node(Id, Props), Dict) :- serialize_node(node(Id, Props), Dict).
serialize_object(edge(F, T, P), Dict) :- serialize_edge(edge(F, T, P), Dict).
serialize_object(position(I, X, Y), Dict) :- serialize_position(position(I, X, Y), Dict).

%% read_mindmap_jsonl(+Stream, -Objects)
%  Read mindmap objects from a JSON Lines stream.
%
read_mindmap_jsonl(Stream, Objects) :-
    read_all_lines(Stream, Lines),
    maplist(json_line_to_dict, Lines, Dicts),
    maplist(deserialize_object, Dicts, Objects).

deserialize_object(Dict, node(Id, Props)) :-
    Dict.type == node, !,
    deserialize_node(Dict, node(Id, Props)).
deserialize_object(Dict, edge(F, T, P)) :-
    Dict.type == edge, !,
    deserialize_edge(Dict, edge(F, T, P)).
deserialize_object(Dict, position(I, X, Y)) :-
    Dict.type == position, !,
    deserialize_position(Dict, position(I, X, Y)).

read_all_lines(Stream, Lines) :-
    read_line_to_string(Stream, Line),
    (   Line == end_of_file
    ->  Lines = []
    ;   Lines = [Line | Rest],
        read_all_lines(Stream, Rest)
    ).

% ============================================================================
% CODE GENERATION FOR OTHER TARGETS
% ============================================================================

%% generate_mindmap_reader(+Target, -Code)
%  Generate code to read mindmap JSON Lines in the target language.
%
generate_mindmap_reader(python, Code) :-
    Code = '
import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class MindmapNode:
    id: str
    label: str
    node_type: str = "default"
    parent: Optional[str] = None
    link: Optional[str] = None
    cluster: Optional[str] = None
    props: Dict[str, Any] = None

@dataclass
class MindmapEdge:
    from_id: str
    to_id: str
    edge_type: str = "default"
    weight: float = 1.0
    label: Optional[str] = None
    props: Dict[str, Any] = None

@dataclass
class MindmapPosition:
    id: str
    x: float
    y: float

def read_mindmap_jsonl(stream=sys.stdin):
    """Read mindmap objects from JSON Lines stream."""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj_type = obj.get("type")

        if obj_type == "node":
            yield MindmapNode(
                id=obj["id"],
                label=obj.get("label", ""),
                node_type=obj.get("node_type", "default"),
                parent=obj.get("parent"),
                link=obj.get("link"),
                cluster=obj.get("cluster"),
                props=obj.get("props", {})
            )
        elif obj_type == "edge":
            yield MindmapEdge(
                from_id=obj["from"],
                to_id=obj["to"],
                edge_type=obj.get("edge_type", "default"),
                weight=obj.get("weight", 1.0),
                label=obj.get("label"),
                props=obj.get("props", {})
            )
        elif obj_type == "position":
            yield MindmapPosition(
                id=obj["id"],
                x=obj["x"],
                y=obj["y"]
            )
'.

generate_mindmap_reader(go, Code) :-
    Code = '
package mindmap

import (
    "bufio"
    "encoding/json"
    "io"
)

type MindmapNode struct {
    ID       string                 `json:"id"`
    Label    string                 `json:"label"`
    NodeType string                 `json:"node_type"`
    Parent   *string                `json:"parent,omitempty"`
    Link     *string                `json:"link,omitempty"`
    Cluster  *string                `json:"cluster,omitempty"`
    Props    map[string]interface{} `json:"props,omitempty"`
}

type MindmapEdge struct {
    From     string                 `json:"from"`
    To       string                 `json:"to"`
    EdgeType string                 `json:"edge_type"`
    Weight   float64                `json:"weight"`
    Label    *string                `json:"label,omitempty"`
    Props    map[string]interface{} `json:"props,omitempty"`
}

type MindmapPosition struct {
    ID string  `json:"id"`
    X  float64 `json:"x"`
    Y  float64 `json:"y"`
}

func ReadMindmapJSONL(reader io.Reader) (<-chan interface{}, <-chan error) {
    objects := make(chan interface{})
    errors := make(chan error, 1)

    go func() {
        defer close(objects)
        defer close(errors)

        scanner := bufio.NewScanner(reader)
        for scanner.Scan() {
            line := scanner.Bytes()
            if len(line) == 0 {
                continue
            }

            var raw map[string]interface{}
            if err := json.Unmarshal(line, &raw); err != nil {
                errors <- err
                return
            }

            switch raw["type"] {
            case "node":
                var node MindmapNode
                json.Unmarshal(line, &node)
                objects <- node
            case "edge":
                var edge MindmapEdge
                json.Unmarshal(line, &edge)
                objects <- edge
            case "position":
                var pos MindmapPosition
                json.Unmarshal(line, &pos)
                objects <- pos
            }
        }
    }()

    return objects, errors
}
'.

generate_mindmap_reader(typescript, Code) :-
    Code = '
interface MindmapNode {
    type: "node";
    id: string;
    label: string;
    node_type: string;
    parent?: string;
    link?: string;
    cluster?: string;
    props?: Record<string, unknown>;
}

interface MindmapEdge {
    type: "edge";
    from: string;
    to: string;
    edge_type: string;
    weight: number;
    label?: string;
    props?: Record<string, unknown>;
}

interface MindmapPosition {
    type: "position";
    id: string;
    x: number;
    y: number;
}

type MindmapObject = MindmapNode | MindmapEdge | MindmapPosition;

function* readMindmapJSONL(lines: string[]): Generator<MindmapObject> {
    for (const line of lines) {
        if (!line.trim()) continue;
        const obj = JSON.parse(line) as MindmapObject;
        yield obj;
    }
}

async function* readMindmapStream(
    reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<MindmapObject> {
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (line.trim()) {
                yield JSON.parse(line) as MindmapObject;
            }
        }
    }
}
'.

%% generate_mindmap_writer(+Target, -Code)
%  Generate code to write mindmap JSON Lines in the target language.
%
generate_mindmap_writer(python, Code) :-
    Code = '
import json
import sys

def write_node(node, stream=sys.stdout):
    """Write a MindmapNode as JSON Line."""
    obj = {
        "type": "node",
        "id": node.id,
        "label": node.label,
        "node_type": node.node_type,
        "parent": node.parent,
        "link": node.link,
        "cluster": node.cluster,
        "props": node.props or {}
    }
    print(json.dumps(obj, ensure_ascii=False), file=stream)

def write_edge(edge, stream=sys.stdout):
    """Write a MindmapEdge as JSON Line."""
    obj = {
        "type": "edge",
        "from": edge.from_id,
        "to": edge.to_id,
        "edge_type": edge.edge_type,
        "weight": edge.weight,
        "label": edge.label,
        "props": edge.props or {}
    }
    print(json.dumps(obj, ensure_ascii=False), file=stream)

def write_position(pos, stream=sys.stdout):
    """Write a MindmapPosition as JSON Line."""
    obj = {
        "type": "position",
        "id": pos.id,
        "x": pos.x,
        "y": pos.y
    }
    print(json.dumps(obj, ensure_ascii=False), file=stream)

def write_positions(positions, stream=sys.stdout):
    """Write position dict {id: (x, y)} as JSON Lines."""
    for node_id, (x, y) in positions.items():
        obj = {"type": "position", "id": node_id, "x": x, "y": y}
        print(json.dumps(obj), file=stream)
'.

generate_mindmap_writer(go, Code) :-
    Code = '
package mindmap

import (
    "encoding/json"
    "io"
)

func WriteNode(w io.Writer, node MindmapNode) error {
    node.Props["type"] = "node"
    return json.NewEncoder(w).Encode(node)
}

func WriteEdge(w io.Writer, edge MindmapEdge) error {
    return json.NewEncoder(w).Encode(struct {
        Type string `json:"type"`
        MindmapEdge
    }{"edge", edge})
}

func WritePosition(w io.Writer, pos MindmapPosition) error {
    return json.NewEncoder(w).Encode(struct {
        Type string `json:"type"`
        MindmapPosition
    }{"position", pos})
}
'.

% ============================================================================
% UTILITIES
% ============================================================================

%% props_to_dict(+Props, -Dict)
%  Convert a list of Prop(Value) terms to a dict.
%
props_to_dict([], json{}).
props_to_dict(Props, Dict) :-
    Props \== [],
    findall(Key-Value, (member(Prop, Props), Prop =.. [Key, Value]), Pairs),
    dict_create(Dict, json, Pairs).

%% dict_to_props(+Dict, -Props)
%  Convert a dict to a list of Prop(Value) terms.
%
dict_to_props(Dict, Props) :-
    dict_pairs(Dict, _, Pairs),
    findall(Prop, (member(Key-Value, Pairs), Prop =.. [Key, Value]), Props).

%% dict_to_json_line(+Dict, -JsonLine)
%  Convert a dict to a JSON string (single line).
%
dict_to_json_line(Dict, JsonLine) :-
    % Use SWI-Prolog's json_write
    with_output_to(string(JsonLine),
        json_write(current_output, Dict, [width(0)])
    ).

%% json_line_to_dict(+JsonLine, -Dict)
%  Parse a JSON string to a dict.
%
json_line_to_dict(JsonLine, Dict) :-
    atom_string(JsonAtom, JsonLine),
    atom_json_dict(JsonAtom, Dict, []).

%% split_json_lines(+JsonL, -Lines)
%  Split JSON Lines string into individual lines.
%
split_json_lines(JsonL, Lines) :-
    split_string(JsonL, "\n", "\r\t ", Lines0),
    exclude(=(""), Lines0, Lines).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_serializer :-
    format('~n=== Mind Map Serializer Tests ===~n~n'),

    % Test 1: Serialize node
    format('Test 1: Serialize node...~n'),
    TestNode = node(n1, [label("Test Node"), type(root), parent(null), link("http://example.com")]),
    serialize_node(TestNode, NodeDict),
    (   NodeDict.id == n1,
        NodeDict.label == "Test Node",
        NodeDict.node_type == root
    ->  format('  PASS: Node serialized correctly~n')
    ;   format('  FAIL: Node serialization incorrect~n')
    ),

    % Test 2: Deserialize node
    format('~nTest 2: Deserialize node...~n'),
    deserialize_node(NodeDict, DeserializedNode),
    DeserializedNode = node(DId, DProps),
    (   DId == n1,
        member(label("Test Node"), DProps)
    ->  format('  PASS: Node deserialized correctly~n')
    ;   format('  FAIL: Node deserialization incorrect~n')
    ),

    % Test 3: Serialize edge
    format('~nTest 3: Serialize edge...~n'),
    TestEdge = edge(n1, n2, [type(strong), weight(2)]),
    serialize_edge(TestEdge, EdgeDict),
    (   EdgeDict.from == n1,
        EdgeDict.to == n2,
        EdgeDict.edge_type == strong
    ->  format('  PASS: Edge serialized correctly~n')
    ;   format('  FAIL: Edge serialization incorrect~n')
    ),

    % Test 4: Serialize position
    format('~nTest 4: Serialize position...~n'),
    TestPos = position(n1, 100.5, 200.3),
    serialize_position(TestPos, PosDict),
    (   PosDict.id == n1,
        PosDict.x =:= 100.5,
        PosDict.y =:= 200.3
    ->  format('  PASS: Position serialized correctly~n')
    ;   format('  FAIL: Position serialization incorrect~n')
    ),

    % Test 5: Batch serialize nodes
    format('~nTest 5: Batch serialize nodes to JSON Lines...~n'),
    Nodes = [
        node(a, [label("A"), type(root)]),
        node(b, [label("B"), type(branch)])
    ],
    serialize_mindmap_nodes(Nodes, JsonL),
    (   sub_string(JsonL, _, _, _, "\"id\":\"a\""),
        sub_string(JsonL, _, _, _, "\"id\":\"b\"")
    ->  format('  PASS: Batch serialization works~n')
    ;   format('  FAIL: Batch serialization failed~n')
    ),

    % Test 6: Generate Python reader
    format('~nTest 6: Generate Python reader code...~n'),
    generate_mindmap_reader(python, PyCode),
    (   sub_string(PyCode, _, _, _, "MindmapNode"),
        sub_string(PyCode, _, _, _, "read_mindmap_jsonl")
    ->  format('  PASS: Python reader generated~n')
    ;   format('  FAIL: Python reader generation failed~n')
    ),

    % Test 7: Protocol registry
    format('~nTest 7: Protocol registry...~n'),
    mindmap_protocol(jsonl, JsonlSpec),
    (   JsonlSpec.supports_streaming == true
    ->  format('  PASS: jsonL protocol registered~n')
    ;   format('  FAIL: Protocol not found~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind map serializer module loaded~n', [])
), now).

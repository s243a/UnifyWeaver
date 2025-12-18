% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% kg_topology.pl - Knowledge Graph Topology for Q-A Systems
%
% Implements Phase 1 of the KG Topology Roadmap:
% - 11 relation types across 3 categories
% - Hash-based anchor linking
% - Seed level provenance tracking
% - Graph traversal API
%
% See: docs/proposals/ROADMAP_KG_TOPOLOGY.md
%      docs/proposals/QA_KNOWLEDGE_GRAPH.md
%      docs/proposals/SEED_QUESTION_TOPOLOGY.md

:- module(kg_topology, [
    % Relation type definitions
    relation_category/2,         % relation_category(?Type, ?Category)
    relation_direction/2,        % relation_direction(?Type, ?Direction)
    valid_relation_type/1,       % valid_relation_type(?Type)

    % Graph traversal API
    get_foundational/3,          % get_foundational(+Config, +AnswerId, -Results)
    get_prerequisites/3,         % get_prerequisites(+Config, +AnswerId, -Results)
    get_extensions/3,            % get_extensions(+Config, +AnswerId, -Results)
    get_next_steps/3,            % get_next_steps(+Config, +AnswerId, -Results)
    get_refined/3,               % get_refined(+Config, +AnswerId, -Results)
    get_general/3,               % get_general(+Config, +AnswerId, -Results)
    get_generalizations/3,       % get_generalizations(+Config, +AnswerId, -Results)
    get_implementations/3,       % get_implementations(+Config, +AnswerId, -Results)
    get_instances/3,             % get_instances(+Config, +AnswerId, -Results)
    get_examples/3,              % get_examples(+Config, +AnswerId, -Results)

    % Anchor linking
    compute_content_hash/2,      % compute_content_hash(+Text, -Hash)
    anchor_question/3,           % anchor_question(+Config, +AnswerId, -QuestionHash)

    % Seed level tracking
    seed_level/3,                % seed_level(+Config, +QuestionId, -Level)
    questions_at_seed_level/4,   % questions_at_seed_level(+Config, +Level, +ClusterId, -Questions)

    % Enhanced search
    search_with_context/5        % search_with_context(+Config, +Query, +TopK, +Options, -Results)
]).

:- use_module(library(lists)).
:- use_module(library(crypto), [crypto_data_hash/3]).
:- use_module(library(option)).

% ============================================================================
% RELATION TYPE DEFINITIONS
% ============================================================================

%% relation_category(?Type, ?Category)
%
%  Categorize relation types into their conceptual groups.
%
%  Categories:
%    - learning_flow: Sequential learning dependencies
%    - scope: Specificity/generality within a domain
%    - abstraction: Abstract patterns vs concrete implementations
%
relation_category(foundational, learning_flow).
relation_category(preliminary, learning_flow).
relation_category(compositional, learning_flow).
relation_category(transitional, learning_flow).

relation_category(refined, scope).
relation_category(general, scope).

relation_category(generalization, abstraction).
relation_category(implementation, abstraction).
relation_category(axiomatization, abstraction).
relation_category(instance, abstraction).
relation_category(example, abstraction).

%% relation_direction(?Type, ?Direction)
%
%  Define the semantic direction of each relation type.
%
%  Directions:
%    - incoming: Target depends on source (A ← B means B depends on A)
%    - outgoing: Source leads to target (A → B means A leads to B)
%
%  Learning flow:
relation_direction(foundational, incoming).   % A is foundational TO B (B depends on A)
relation_direction(preliminary, incoming).    % A is preliminary TO B (B requires A first)
relation_direction(compositional, outgoing).  % A composes INTO B (B extends A)
relation_direction(transitional, outgoing).   % A transitions TO B (B follows A)

%  Scope:
relation_direction(refined, outgoing).        % A refines INTO B (B is more specific)
relation_direction(general, incoming).        % A is general FOR B (B is more specific)

%  Abstraction:
relation_direction(generalization, outgoing). % A generalizes INTO B (B is pattern)
relation_direction(implementation, incoming). % A implements B (A is code for B)
relation_direction(axiomatization, outgoing). % A axiomatizes INTO B (B is theory)
relation_direction(instance, incoming).       % A is instance OF B (A satisfies B)
relation_direction(example, incoming).        % A is example OF B (A demonstrates B)

%% valid_relation_type(?Type)
%
%  Check if a relation type is valid.
%
valid_relation_type(Type) :-
    relation_category(Type, _).

% ============================================================================
% GRAPH TRAVERSAL API
% ============================================================================

%% get_foundational(+Config, +AnswerId, -Results)
%
%  Get concepts that this answer depends on (foundational knowledge).
%
get_foundational(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, foundational, incoming, Results).

%% get_prerequisites(+Config, +AnswerId, -Results)
%
%  Get practical prerequisites (setup steps) required before this answer.
%
get_prerequisites(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, preliminary, incoming, Results).

%% get_extensions(+Config, +AnswerId, -Results)
%
%  Get answers that extend or build upon this one.
%
get_extensions(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, compositional, outgoing, Results).

%% get_next_steps(+Config, +AnswerId, -Results)
%
%  Get natural next steps after this answer.
%
get_next_steps(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, transitional, outgoing, Results).

%% get_refined(+Config, +AnswerId, -Results)
%
%  Get more specific variants of this answer.
%
get_refined(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, refined, outgoing, Results).

%% get_general(+Config, +AnswerId, -Results)
%
%  Get broader/more general versions of this answer.
%
get_general(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, general, incoming, Results).

%% get_generalizations(+Config, +AnswerId, -Results)
%
%  Get abstract patterns derived from this answer.
%
get_generalizations(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, generalization, outgoing, Results).

%% get_implementations(+Config, +AnswerId, -Results)
%
%  Get code implementations of this pattern.
%
get_implementations(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, implementation, incoming, Results).

%% get_instances(+Config, +AnswerId, -Results)
%
%  Get domain instances that satisfy this theory.
%
get_instances(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, instance, incoming, Results).

%% get_examples(+Config, +AnswerId, -Results)
%
%  Get pedagogical examples that demonstrate this concept.
%
get_examples(Config, AnswerId, Results) :-
    get_relations(Config, AnswerId, example, incoming, Results).

%% get_relations(+Config, +AnswerId, +RelationType, +Direction, -Results)
%
%  Generic relation lookup via Python backend.
%
get_relations(Config, AnswerId, RelationType, Direction, Results) :-
    member(db_path(DbPath), Config),
    format(atom(PythonCode),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from kg_topology_api import KGTopologyAPI
import json

api = KGTopologyAPI("~w")
results = api.get_relations(~w, "~w", "~w")
print(json.dumps(results))
api.close()
', [DbPath, AnswerId, RelationType, Direction]),
    run_python_kg(PythonCode, Results).

% ============================================================================
% ANCHOR LINKING
% ============================================================================

%% compute_content_hash(+Text, -Hash)
%
%  Compute SHA-256 hash of content for anchor linking.
%
compute_content_hash(Text, Hash) :-
    (   atom(Text)
    ->  atom_string(Text, TextStr)
    ;   TextStr = Text
    ),
    crypto_data_hash(TextStr, Hash, [algorithm(sha256), encoding(hex)]).

%% anchor_question(+Config, +AnswerId, -QuestionHash)
%
%  Get the anchor question hash for an answer.
%  The anchor question is the original question that generated this answer.
%
anchor_question(Config, AnswerId, QuestionHash) :-
    member(db_path(DbPath), Config),
    format(atom(PythonCode),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from kg_topology_api import KGTopologyAPI

api = KGTopologyAPI("~w")
result = api.get_anchor_question(~w)
if result:
    print(result)
else:
    print("none")
api.close()
', [DbPath, AnswerId]),
    run_python_kg_simple(PythonCode, Result),
    (   Result = "none"
    ->  fail
    ;   QuestionHash = Result
    ).

% ============================================================================
% SEED LEVEL TRACKING
% ============================================================================

%% seed_level(+Config, +QuestionId, -Level)
%
%  Get the seed level of a question.
%  - seed(0): Original curated dataset
%  - seed(n): Questions discovered at expansion depth n
%
seed_level(Config, QuestionId, Level) :-
    member(db_path(DbPath), Config),
    format(atom(PythonCode),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from kg_topology_api import KGTopologyAPI

api = KGTopologyAPI("~w")
level = api.get_seed_level(~w)
print(level if level is not None else -1)
api.close()
', [DbPath, QuestionId]),
    run_python_kg_simple(PythonCode, LevelStr),
    atom_number(LevelStr, Level),
    Level >= 0.

%% questions_at_seed_level(+Config, +Level, +ClusterId, -Questions)
%
%  Get all questions at a specific seed level within a cluster.
%  Useful for diversity when expanding clusters.
%
questions_at_seed_level(Config, Level, ClusterId, Questions) :-
    member(db_path(DbPath), Config),
    format(atom(PythonCode),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from kg_topology_api import KGTopologyAPI
import json

api = KGTopologyAPI("~w")
questions = api.get_questions_at_seed_level(~w, ~w)
print(json.dumps(questions))
api.close()
', [DbPath, Level, ClusterId]),
    run_python_kg(PythonCode, Questions).

% ============================================================================
% ENHANCED SEARCH
% ============================================================================

%% search_with_context(+Config, +Query, +TopK, +Options, -Results)
%
%  Semantic search with knowledge graph context.
%  Returns results enriched with related answers.
%
%  Options:
%    - include_foundational(Bool): Include foundational concepts
%    - include_prerequisites(Bool): Include prerequisites
%    - include_extensions(Bool): Include extensions
%    - include_next_steps(Bool): Include next steps
%    - context_depth(N): How many hops to traverse (default: 1)
%
search_with_context(Config, Query, TopK, Options, Results) :-
    member(db_path(DbPath), Config),
    % Get model name
    (   member(model_name(ModelName), Config)
    ->  true
    ;   ModelName = 'all-MiniLM-L6-v2'
    ),
    % Get projection ID
    (   member(mh_projection_id(ProjId), Config)
    ->  true
    ;   member(projection_id(ProjId), Config)
    ->  true
    ;   ProjId = 1
    ),
    % Get options
    option(include_foundational(InclFound), Options, true),
    option(include_prerequisites(InclPre), Options, true),
    option(include_extensions(InclExt), Options, true),
    option(include_next_steps(InclNext), Options, true),
    option(context_depth(Depth), Options, 1),
    % Escape query
    escape_for_python_kg(Query, EscapedQuery),
    % Build Python code
    format(atom(PythonCode),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from kg_topology_api import KGTopologyAPI
import json

api = KGTopologyAPI("~w")
results = api.search_with_context(
    query_text="~w",
    model_name="~w",
    projection_id=~w,
    top_k=~w,
    include_foundational=~w,
    include_prerequisites=~w,
    include_extensions=~w,
    include_next_steps=~w,
    context_depth=~w
)
print(json.dumps(results))
api.close()
', [DbPath, EscapedQuery, ModelName, ProjId, TopK,
    InclFound, InclPre, InclExt, InclNext, Depth]),
    run_python_kg(PythonCode, Results).

% ============================================================================
% PYTHON INTERFACE UTILITIES
% ============================================================================

%% run_python_kg(+Code, -Results)
%
%  Execute Python code and parse JSON results.
%
run_python_kg(Code, Results) :-
    process_create(path(python3), ['-c', Code],
                  [stdout(pipe(Out)), stderr(pipe(Err)), process(Proc)]),
    read_string(Out, _, OutputStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Proc, Status),
    (   Status = exit(0)
    ->  parse_json_kg(OutputStr, Results)
    ;   format('KG Topology error: ~w~n', [ErrStr]),
        Results = []
    ).

%% run_python_kg_simple(+Code, -Result)
%
%  Execute Python code and return simple string result.
%
run_python_kg_simple(Code, Result) :-
    process_create(path(python3), ['-c', Code],
                  [stdout(pipe(Out)), stderr(null), process(Proc)]),
    read_string(Out, _, OutputStr),
    close(Out),
    process_wait(Proc, _),
    string_codes(OutputStr, Codes),
    exclude(=(0'\n), Codes, CleanCodes),
    string_codes(Result, CleanCodes).

%% parse_json_kg(+JsonStr, -Results)
%
%  Parse JSON results into Prolog terms.
%
parse_json_kg(JsonStr, Results) :-
    % Use Python to convert JSON to Prolog format
    format(atom(Code),
'import json
data = json.loads("""~w""")
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            pairs = ", ".join([f"{k}({repr(v)})" for k, v in item.items()])
            print(f"[{pairs}].")
        else:
            print(f"{repr(item)}.")
else:
    print(f"{repr(data)}.")
', [JsonStr]),
    process_create(path(python3), ['-c', Code],
                  [stdout(pipe(Out)), stderr(null), process(Proc)]),
    read_string(Out, _, OutputStr),
    close(Out),
    process_wait(Proc, _),
    % Parse each line as a Prolog term
    split_string(OutputStr, "\n", "\n", Lines),
    findall(Result,
            (member(Line, Lines),
             Line \= "",
             atom_string(LineAtom, Line),
             catch(read_term_from_atom(LineAtom, Result, []), _, fail)),
            Results).

%% escape_for_python_kg(+Input, -Output)
%
%  Escape string for embedding in Python code.
%
escape_for_python_kg(Input, Output) :-
    (   atom(Input) -> atom_string(Input, Str) ; Str = Input ),
    string_codes(Str, Codes),
    escape_codes_kg(Codes, EscapedCodes),
    string_codes(Output, EscapedCodes).

escape_codes_kg([], []).
escape_codes_kg([C|Cs], Escaped) :-
    escape_code_kg(C, EscapedC),
    append(EscapedC, RestEscaped, Escaped),
    escape_codes_kg(Cs, RestEscaped).

escape_code_kg(0'\\, [0'\\, 0'\\]).
escape_code_kg(0'\", [0'\\, 0'\"]).
escape_code_kg(0'\n, [0'\\, 0'n]).
escape_code_kg(0'\r, [0'\\, 0'r]).
escape_code_kg(0'\t, [0'\\, 0't]).
escape_code_kg(C, [C]) :- C \= 0'\\, C \= 0'\", C \= 0'\n, C \= 0'\r, C \= 0'\t.

% ============================================================================
% REGISTRATION
% ============================================================================

:- use_module('../core/component_registry', [
    register_component_type/4
]).

:- initialization((
    register_component_type(runtime, kg_topology, kg_topology, [
        description("Knowledge graph topology for Q-A systems")
    ])
), now).

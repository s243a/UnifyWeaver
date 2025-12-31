% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% semantic_search.pl - Semantic Search Component with LDA Projection
%
% Provides semantic search over playbook examples and documentation
% using multi-head LDA projection for improved retrieval.
%
% See: docs/proposals/SEMANTIC_PROJECTION_LDA.md
%      docs/proposals/MULTI_HEAD_PROJECTION_THEORY.md

:- module(semantic_search, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    shutdown_component/1,

    % High-level API
    find_examples/3,        % +Query, +TopK, -Examples
    find_examples/4,        % +Query, +TopK, +Options, -Examples
    semantic_search/3,      % +Query, +TopK, -Results
    semantic_search/4       % +Query, +TopK, +Options, -Results
]).

:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(option)).

% ============================================================================
% TYPE INFO
% ============================================================================

%% type_info(-Info)
%
%  Metadata about this component type.
%
type_info(info(
    name('Semantic Search'),
    version('1.0.0'),
    description('Semantic search with multi-head LDA projection'),
    default_backend(python),
    supported_backends([python]),  % Future: go
    requires([numpy, 'sentence-transformers']),
    config_options([
        db_path - 'Path to LDA SQLite database',
        model_name - 'Embedding model name (default: all-MiniLM-L6-v2)',
        mh_projection_id - 'Multi-head projection ID to use (optional)',
        projection_id - 'Global projection ID to use (optional, fallback)',
        use_projection - 'Whether to use projection (default: true)',
        temperature - 'Softmax temperature for multi-head routing (default: 0.1)'
    ])
)).

% ============================================================================
% DYNAMIC STATE
% ============================================================================

:- dynamic search_config/2.  % search_config(Name, Config)

% ============================================================================
% CONFIGURATION VALIDATION
% ============================================================================

%% validate_config(+Config)
%
%  Validate component configuration.
%
validate_config(Config) :-
    % Required: db_path
    (   member(db_path(Path), Config)
    ->  atom(Path)
    ;   throw(error(missing_required_option(db_path), context(validate_config/1, _)))
    ),
    % Optional: model_name (atom)
    (   member(model_name(M), Config)
    ->  atom(M)
    ;   true
    ),
    % Optional: mh_projection_id (integer)
    (   member(mh_projection_id(MH), Config)
    ->  integer(MH), MH > 0
    ;   true
    ),
    % Optional: projection_id (integer)
    (   member(projection_id(P), Config)
    ->  integer(P), P > 0
    ;   true
    ),
    % Optional: temperature (number)
    (   member(temperature(T), Config)
    ->  number(T), T > 0
    ;   true
    ).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_component(+Name, +Config)
%
%  Initialize the semantic search component.
%
init_component(Name, Config) :-
    member(db_path(DbPath), Config),
    (   exists_file(DbPath)
    ->  format('Semantic search ~w: Database verified: ~w~n', [Name, DbPath]),
        % Store config for later use
        retractall(search_config(Name, _)),
        assertz(search_config(Name, Config)),
        % Verify database is readable
        (   verify_database(DbPath)
        ->  format('Semantic search ~w: Database is accessible~n', [Name])
        ;   format('Semantic search ~w: Warning - could not verify database~n', [Name])
        )
    ;   format('Semantic search ~w: Error - database not found: ~w~n', [Name, DbPath]),
        throw(error(db_file_not_found(DbPath), context(init_component/2, _)))
    ).

%% verify_database(+DbPath)
%
%  Verify that the database is accessible.
%
verify_database(DbPath) :-
    format(atom(PythonCode),
'import sqlite3
try:
    conn = sqlite3.connect("~w")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM answers")
    count = cur.fetchone()[0]
    conn.close()
    print(f"OK:{count}")
except Exception as e:
    print(f"ERROR:{e}")
', [DbPath]),
    catch(
        (   process_create(path(python3), ['-c', PythonCode],
                          [stdout(pipe(Out)), stderr(null), process(Proc)]),
            read_string(Out, _, Result),
            close(Out),
            process_wait(Proc, exit(0)),
            sub_string(Result, 0, 2, _, "OK")
        ),
        _,
        fail
    ).

% ============================================================================
% INVOCATION
% ============================================================================

%% invoke_component(+Name, +Config, +Input, -Output)
%
%  Invoke semantic search.
%
%  Input formats:
%    - search(Query, TopK) : Search for top K results
%    - search(Query, TopK, Options) : Search with options
%
%  Output formats:
%    - results(List) : List of result dicts
%
invoke_component(_Name, Config, search(Query, TopK), results(Results)) :-
    semantic_search_impl(Config, Query, TopK, [], Results).

invoke_component(_Name, Config, search(Query, TopK, Options), results(Results)) :-
    semantic_search_impl(Config, Query, TopK, Options, Results).

% ============================================================================
% HIGH-LEVEL API
% ============================================================================

%% find_examples(+Query, +TopK, -Examples)
%
%  Find playbook examples matching the query.
%  Uses the default semantic search component.
%
%  @param Query   atom or string - natural language query
%  @param TopK    integer - number of results to return
%  @param Examples list - list of example dicts with id, score, text
%
find_examples(Query, TopK, Examples) :-
    find_examples(Query, TopK, [], Examples).

%% find_examples(+Query, +TopK, +Options, -Examples)
%
%  Find playbook examples with options.
%
%  Options:
%    - component(Name) : Use specific component instance
%    - use_projection(Bool) : Whether to use LDA projection
%    - min_score(Score) : Minimum score threshold
%
find_examples(Query, TopK, Options, Examples) :-
    % Get component config
    (   option(component(Name), Options)
    ->  true
    ;   Name = default_search
    ),
    (   search_config(Name, Config)
    ->  true
    ;   % Try to use default config from environment or fallback
        default_search_config(Config)
    ),
    semantic_search_impl(Config, Query, TopK, Options, Results),
    % Filter by min_score if specified
    (   option(min_score(MinScore), Options)
    ->  include(result_above_score(MinScore), Results, Examples)
    ;   Examples = Results
    ).

result_above_score(MinScore, Result) :-
    member(score(Score), Result),
    Score >= MinScore.

%% semantic_search(+Query, +TopK, -Results)
%
%  Perform semantic search.
%
semantic_search(Query, TopK, Results) :-
    semantic_search(Query, TopK, [], Results).

%% semantic_search(+Query, +TopK, +Options, -Results)
%
%  Perform semantic search with options.
%
semantic_search(Query, TopK, Options, Results) :-
    (   option(component(Name), Options)
    ->  true
    ;   Name = default_search
    ),
    (   search_config(Name, Config)
    ->  true
    ;   default_search_config(Config)
    ),
    semantic_search_impl(Config, Query, TopK, Options, Results).

% ============================================================================
% IMPLEMENTATION
% ============================================================================

%% semantic_search_impl(+Config, +Query, +TopK, +Options, -Results)
%
%  Implementation of semantic search using Python subprocess.
%
semantic_search_impl(Config, Query, TopK, Options, Results) :-
    member(db_path(DbPath), Config),
    % Get model name
    (   member(model_name(ModelName), Config)
    ->  true
    ;   ModelName = 'all-MiniLM-L6-v2'
    ),
    % Determine search mode
    (   option(use_projection(false), Options)
    ->  SearchMode = direct
    ;   member(mh_projection_id(MhId), Config)
    ->  SearchMode = multi_head(MhId)
    ;   member(projection_id(ProjId), Config)
    ->  SearchMode = global(ProjId)
    ;   SearchMode = direct
    ),
    % Escape query for Python
    escape_for_python(Query, EscapedQuery),
    % Build Python code based on search mode
    build_search_code(DbPath, ModelName, EscapedQuery, TopK, SearchMode, PythonCode),
    % Run search
    run_python_search(PythonCode, Results).

%% build_search_code(+DbPath, +ModelName, +Query, +TopK, +Mode, -Code)
%
%  Build Python code for the search.
%
build_search_code(DbPath, ModelName, Query, TopK, direct, Code) :-
    format(atom(Code),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from lda_database import LDAProjectionDB
from sentence_transformers import SentenceTransformer
import json
import numpy as np

db = LDAProjectionDB("~w")
model = SentenceTransformer("~w")

# Embed query
query_emb = model.encode("~w", convert_to_numpy=True)
query_norm = np.linalg.norm(query_emb)
if query_norm > 0:
    query_emb = query_emb / query_norm

# Get all answer embeddings
model_info = db.get_model("~w")
if not model_info:
    print("[]")
    sys.exit(0)

model_id = model_info["model_id"]
answer_ids, answer_matrix = db.get_all_answer_embeddings(model_id)

if len(answer_ids) == 0:
    print("[]")
    sys.exit(0)

# Normalize
answer_norms = np.linalg.norm(answer_matrix, axis=1, keepdims=True)
answer_norms = np.where(answer_norms > 0, answer_norms, 1)
answer_matrix_normed = answer_matrix / answer_norms

# Compute similarities
scores = answer_matrix_normed @ query_emb
top_indices = np.argsort(-scores)[:~w]

# Build results
results = []
for idx in top_indices:
    answer_id = answer_ids[idx]
    answer = db.get_answer(answer_id)
    results.append({
        "answer_id": answer_id,
        "score": float(scores[idx]),
        "text": answer["text"][:500],
        "record_id": answer.get("record_id", ""),
        "source_file": answer.get("source_file", "")
    })

print(json.dumps(results))
db.close()
', [DbPath, ModelName, Query, ModelName, TopK]).

build_search_code(DbPath, ModelName, Query, TopK, multi_head(MhId), Code) :-
    format(atom(Code),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from lda_database import LDAProjectionDB
from sentence_transformers import SentenceTransformer
import json

db = LDAProjectionDB("~w")
model = SentenceTransformer("~w")

# Embed query
query_emb = model.encode("~w", convert_to_numpy=True)

# Multi-head search
results = db.multi_head_search(
    query_embedding=query_emb,
    mh_projection_id=~w,
    top_k=~w,
    log=False
)

# Format results
output = []
for r in results:
    output.append({
        "answer_id": r["answer_id"],
        "score": r["score"],
        "text": r["text"][:500],
        "record_id": r.get("record_id", ""),
        "source_file": r.get("source_file", ""),
        "routing_weights": r.get("routing_weights", {})
    })

print(json.dumps(output))
db.close()
', [DbPath, ModelName, Query, MhId, TopK]).

build_search_code(DbPath, ModelName, Query, TopK, global(ProjId), Code) :-
    format(atom(Code),
'import sys
sys.path.insert(0, "src/unifyweaver/targets/python_runtime")
from lda_database import LDAProjectionDB
from sentence_transformers import SentenceTransformer
import json

db = LDAProjectionDB("~w")
model = SentenceTransformer("~w")

# Embed query
query_emb = model.encode("~w", convert_to_numpy=True)

# Search with projection
results = db.search(
    query_embedding=query_emb,
    projection_id=~w,
    top_k=~w,
    log=False
)

# Format results
output = []
for r in results:
    output.append({
        "answer_id": r["answer_id"],
        "score": r["score"],
        "text": r["text"][:500],
        "record_id": r.get("record_id", ""),
        "source_file": r.get("source_file", "")
    })

print(json.dumps(output))
db.close()
', [DbPath, ModelName, Query, ProjId, TopK]).

%% run_python_search(+Code, -Results)
%
%  Execute Python search code and parse results.
%
run_python_search(Code, Results) :-
    process_create(path(python3), ['-c', Code],
                  [stdout(pipe(Out)), stderr(pipe(Err)), process(Proc)]),
    read_string(Out, _, OutputStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Proc, Status),
    (   Status = exit(0)
    ->  parse_json_results(OutputStr, Results)
    ;   format('Semantic search error: ~w~n', [ErrStr]),
        Results = []
    ).

%% parse_json_results(+JsonStr, -Results)
%
%  Parse JSON results into Prolog list.
%
parse_json_results(JsonStr, Results) :-
    % Use Python to convert JSON to Prolog-readable format
    format(atom(Code),
'import json
data = json.loads("""~w""")
for item in data:
    print("[", end="")
    print(f"answer_id({item[\"answer_id\"]}),", end="")
    print(f"score({item[\"score\"]}),", end="")
    # Escape text for Prolog
    text = item["text"].replace("\\\\", "\\\\\\\\").replace("\"", "\\\\\"").replace("\\n", " ")
    print(f"text(\\"{text}\\"),", end="")
    rid = item.get("record_id", "")
    print(f"record_id(\\"{rid}\\"),", end="")
    sf = item.get("source_file", "")
    print(f"source_file(\\"{sf}\\")", end="")
    print("].")
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

%% escape_for_python(+Input, -Output)
%
%  Escape string for embedding in Python code.
%
escape_for_python(Input, Output) :-
    (   atom(Input) -> atom_string(Input, Str) ; Str = Input ),
    string_codes(Str, Codes),
    escape_codes(Codes, EscapedCodes),
    string_codes(Output, EscapedCodes).

escape_codes([], []).
escape_codes([C|Cs], Escaped) :-
    escape_code(C, EscapedC),
    append(EscapedC, RestEscaped, Escaped),
    escape_codes(Cs, RestEscaped).

% Escape special characters
escape_code(0'\\, [0'\\, 0'\\]).
escape_code(0'\", [0'\\, 0'\"]).
escape_code(0'\n, [0'\\, 0'n]).
escape_code(0'\r, [0'\\, 0'r]).
escape_code(0'\t, [0'\\, 0't]).
escape_code(C, [C]) :- C \= 0'\\, C \= 0'\", C \= 0'\n, C \= 0'\r, C \= 0'\t.

% ============================================================================
% DEFAULT CONFIGURATION
% ============================================================================

%% default_search_config(-Config)
%
%  Get default search configuration.
%
default_search_config(Config) :-
    % Try to find default database
    (   exists_file('playbooks/lda-training-data/lda.db')
    ->  DbPath = 'playbooks/lda-training-data/lda.db'
    ;   DbPath = 'lda.db'
    ),
    Config = [
        db_path(DbPath),
        model_name('all-MiniLM-L6-v2'),
        mh_projection_id(2)  % Use multi-head with temp=0.1
    ].

% ============================================================================
% SHUTDOWN
% ============================================================================

%% shutdown_component(+Name)
%
%  Shutdown the component.
%
shutdown_component(Name) :-
    retractall(search_config(Name, _)),
    format('Semantic search ~w: Shutdown~n', [Name]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- use_module('../core/component_registry', [
    register_component_type/4
]).

:- initialization((
    register_component_type(runtime, semantic_search, semantic_search, [
        description("Semantic search with multi-head LDA projection")
    ])
), now).

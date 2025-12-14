% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% lda_projection.pl - LDA-based Semantic Projection Component
%
% Projects query embeddings to answer space using a learned transformation
% matrix W derived from Q-A pairs via Linear Discriminant Analysis.
%
% See: docs/proposals/SEMANTIC_PROJECTION_LDA.md
%      docs/proposals/COMPONENT_REGISTRY.md

:- module(lda_projection, [
    type_info/1,
    validate_config/1,
    init_component/2,
    invoke_component/4,
    shutdown_component/1
]).

:- use_module(library(lists)).
:- use_module(library(process)).

% ============================================================================
% TYPE INFO
% ============================================================================

%% type_info(-Info)
%
%  Metadata about this component type.
%
type_info(info(
    name('LDA Semantic Projection'),
    version('1.0.0'),
    description('Projects query embeddings to answer space using learned W matrix'),
    default_backend(python),
    supported_backends([python]),  % Future: go, rust
    requires([numpy]),
    config_options([
        model_file - 'Path to W matrix file (.npy or .json)',
        embedding_dim - 'Embedding dimension (default: 384)',
        backend - 'Execution backend: python (default), go, rust (future)',
        lambda_reg - 'Regularization parameter used in training (informational)',
        ridge - 'Ridge parameter used in training (informational)'
    ])
)).

% ============================================================================
% CONFIGURATION VALIDATION
% ============================================================================

%% validate_config(+Config)
%
%  Validate component configuration.
%
validate_config(Config) :-
    % Required: model_file
    (   member(model_file(File), Config)
    ->  atom(File)
    ;   throw(error(missing_required_option(model_file), context(validate_config/1, _)))
    ),
    % Optional: backend (default: python)
    (   member(backend(Backend), Config)
    ->  (   member(Backend, [python])  % Supported backends
        ->  true
        ;   throw(error(unsupported_backend(Backend), context(validate_config/1, _)))
        )
    ;   true  % Default to python
    ),
    % Optional: embedding_dim (integer)
    (   member(embedding_dim(D), Config)
    ->  integer(D), D > 0
    ;   true
    ),
    % Optional: lambda_reg (number)
    (   member(lambda_reg(L), Config)
    ->  number(L), L >= 0
    ;   true
    ),
    % Optional: ridge (number)
    (   member(ridge(R), Config)
    ->  number(R), R >= 0
    ;   true
    ).

%% get_backend(+Config, -Backend)
%
%  Get the backend from config, defaulting to python.
%
get_backend(Config, Backend) :-
    (   member(backend(B), Config)
    ->  Backend = B
    ;   Backend = python
    ).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_component(+Name, +Config)
%
%  Initialize the LDA projection component.
%  Verifies the model file exists and can be loaded.
%
init_component(Name, Config) :-
    member(model_file(File), Config),
    (   exists_file(File)
    ->  format('LDA projection ~w: Model file verified: ~w~n', [Name, File]),
        % Optionally verify we can load it via Python
        (   verify_model_loadable(File)
        ->  format('LDA projection ~w: Model is loadable~n', [Name])
        ;   format('LDA projection ~w: Warning - could not verify model loadability~n', [Name])
        )
    ;   format('LDA projection ~w: Error - model file not found: ~w~n', [Name, File]),
        throw(error(model_file_not_found(File), context(init_component/2, _)))
    ).

%% verify_model_loadable(+File)
%
%  Verify that Python can load the model file.
%
verify_model_loadable(File) :-
    % Try to load via Python subprocess
    format(atom(PythonCode),
'import sys
try:
    import numpy as np
    if "~w".endswith(".npy"):
        W = np.load("~w")
    else:
        import json
        with open("~w") as f:
            W = np.array(json.load(f))
    print("OK:" + str(W.shape))
except Exception as e:
    print("ERROR:" + str(e))
    sys.exit(1)
', [File, File, File]),
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
%  Invoke the LDA projection.
%
%  Input formats:
%    - query(Embedding) : Project single embedding (list of floats)
%    - query_batch(Embeddings) : Project batch of embeddings
%    - similarity(QueryEmb, DocEmb) : Compute projected similarity
%
%  Output formats:
%    - projected(Embedding) : Projected embedding
%    - projected_batch(Embeddings) : Projected embeddings
%    - score(Score) : Similarity score
%
invoke_component(_Name, Config, query(QueryEmbedding), projected(ProjectedEmbedding)) :-
    member(model_file(File), Config),
    get_backend(Config, Backend),
    project_embedding(Backend, File, QueryEmbedding, ProjectedEmbedding).

invoke_component(_Name, Config, query_batch(QueryEmbeddings), projected_batch(ProjectedEmbeddings)) :-
    member(model_file(File), Config),
    get_backend(Config, Backend),
    project_embeddings_batch(Backend, File, QueryEmbeddings, ProjectedEmbeddings).

invoke_component(_Name, Config, similarity(QueryEmb, DocEmb), score(Score)) :-
    member(model_file(File), Config),
    get_backend(Config, Backend),
    compute_projected_similarity(Backend, File, QueryEmb, DocEmb, Score).

% ============================================================================
% BACKEND DISPATCH
% ============================================================================

%% project_embedding(+Backend, +ModelFile, +QueryEmbedding, -ProjectedEmbedding)
%
%  Project a single embedding using the specified backend.
%
project_embedding(python, ModelFile, QueryEmbedding, ProjectedEmbedding) :-
    project_embedding_python(ModelFile, QueryEmbedding, ProjectedEmbedding).
% Future: project_embedding(go, ModelFile, QueryEmbedding, ProjectedEmbedding) :- ...
% Future: project_embedding(rust, ModelFile, QueryEmbedding, ProjectedEmbedding) :- ...

%% project_embeddings_batch(+Backend, +ModelFile, +QueryEmbeddings, -ProjectedEmbeddings)
%
%  Project a batch of embeddings using the specified backend.
%
project_embeddings_batch(python, ModelFile, QueryEmbeddings, ProjectedEmbeddings) :-
    project_embeddings_batch_python(ModelFile, QueryEmbeddings, ProjectedEmbeddings).

%% compute_projected_similarity(+Backend, +ModelFile, +QueryEmb, +DocEmb, -Score)
%
%  Compute projected similarity using the specified backend.
%
compute_projected_similarity(python, ModelFile, QueryEmb, DocEmb, Score) :-
    compute_projected_similarity_python(ModelFile, QueryEmb, DocEmb, Score).

% ============================================================================
% PYTHON BACKEND
% ============================================================================

%% project_embedding_python(+ModelFile, +QueryEmbedding, -ProjectedEmbedding)
%
%  Project a single embedding using Python.
%
project_embedding_python(ModelFile, QueryEmbedding, ProjectedEmbedding) :-
    embedding_to_json(QueryEmbedding, QueryJson),
    format(atom(PythonCode),
'import numpy as np
import json
import sys

# Load model
if "~w".endswith(".npy"):
    W = np.load("~w")
else:
    with open("~w") as f:
        W = np.array(json.load(f))

# Load query
query = np.array(~w)

# Project
projected = W @ query

# Output
print(json.dumps(projected.tolist()))
', [ModelFile, ModelFile, ModelFile, QueryJson]),
    run_python_code(PythonCode, ResultJson),
    json_to_embedding(ResultJson, ProjectedEmbedding).

%% project_embeddings_batch_python(+ModelFile, +QueryEmbeddings, -ProjectedEmbeddings)
%
%  Project a batch of embeddings using Python.
%
project_embeddings_batch_python(ModelFile, QueryEmbeddings, ProjectedEmbeddings) :-
    embeddings_to_json(QueryEmbeddings, QueriesJson),
    format(atom(PythonCode),
'import numpy as np
import json
import sys

# Load model
if "~w".endswith(".npy"):
    W = np.load("~w")
else:
    with open("~w") as f:
        W = np.array(json.load(f))

# Load queries
queries = np.array(~w)

# Project (queries @ W.T for batch)
projected = queries @ W.T

# Output
print(json.dumps(projected.tolist()))
', [ModelFile, ModelFile, ModelFile, QueriesJson]),
    run_python_code(PythonCode, ResultJson),
    json_to_embeddings(ResultJson, ProjectedEmbeddings).

%% compute_projected_similarity_python(+ModelFile, +QueryEmb, +DocEmb, -Score)
%
%  Compute cosine similarity between projected query and document.
%
compute_projected_similarity_python(ModelFile, QueryEmb, DocEmb, Score) :-
    embedding_to_json(QueryEmb, QueryJson),
    embedding_to_json(DocEmb, DocJson),
    format(atom(PythonCode),
'import numpy as np
import json

# Load model
if "~w".endswith(".npy"):
    W = np.load("~w")
else:
    with open("~w") as f:
        W = np.array(json.load(f))

# Load embeddings
query = np.array(~w)
doc = np.array(~w)

# Project query
projected = W @ query

# Cosine similarity
score = float(np.dot(projected, doc) / (np.linalg.norm(projected) * np.linalg.norm(doc)))
print(score)
', [ModelFile, ModelFile, ModelFile, QueryJson, DocJson]),
    run_python_code(PythonCode, ScoreStr),
    atom_string(ScoreAtom, ScoreStr),
    atom_number(ScoreAtom, Score).

% ============================================================================
% UTILITIES
% ============================================================================

%% run_python_code(+Code, -Output)
%
%  Execute Python code and capture stdout.
%
run_python_code(Code, Output) :-
    process_create(path(python3), ['-c', Code],
                  [stdout(pipe(Out)), stderr(pipe(Err)), process(Proc)]),
    read_string(Out, _, OutputRaw),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Proc, Status),
    (   Status = exit(0)
    ->  % Trim whitespace
        string_codes(OutputRaw, Codes),
        exclude(is_whitespace_code, Codes, TrimmedCodes),
        string_codes(Output, TrimmedCodes)
    ;   format('Python error: ~w~n', [ErrStr]),
        fail
    ).

is_whitespace_code(32).   % space
is_whitespace_code(9).    % tab
is_whitespace_code(10).   % newline
is_whitespace_code(13).   % carriage return

%% embedding_to_json(+Embedding, -Json)
%
%  Convert embedding list to JSON string.
%
embedding_to_json(Embedding, Json) :-
    format(atom(Json), '~w', [Embedding]).

%% embeddings_to_json(+Embeddings, -Json)
%
%  Convert list of embeddings to JSON string.
%
embeddings_to_json(Embeddings, Json) :-
    format(atom(Json), '~w', [Embeddings]).

%% json_to_embedding(+Json, -Embedding)
%
%  Parse JSON string to embedding list.
%
json_to_embedding(Json, Embedding) :-
    atom_string(JsonAtom, Json),
    read_term_from_atom(JsonAtom, Embedding, []).

%% json_to_embeddings(+Json, -Embeddings)
%
%  Parse JSON string to list of embeddings.
%
json_to_embeddings(Json, Embeddings) :-
    atom_string(JsonAtom, Json),
    read_term_from_atom(JsonAtom, Embeddings, []).

% ============================================================================
% SHUTDOWN
% ============================================================================

%% shutdown_component(+Name)
%
%  Shutdown the component (cleanup if needed).
%
shutdown_component(Name) :-
    format('LDA projection ~w: Shutdown~n', [Name]).

% ============================================================================
% REGISTRATION
% ============================================================================

:- use_module('../core/component_registry', [
    register_component_type/4
]).

:- initialization((
    register_component_type(runtime, lda_projection, lda_projection, [
        description("LDA-based semantic projection for RAG queries")
    ])
), now).

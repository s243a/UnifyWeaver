:- module(semantic_source, []).

%% Semantic/Vector Search Source Plugin
%
% Provides semantic search capabilities via embedding vectors and similarity search.
%
% This plugin enables semantic retrieval over document collections with:
% - Multiple target wrappers: bash, powershell, csharp, python
% - Multiple embedding backends: python_onnx, go_service, rust_candle, csharp_native
%
% Example usage:
%   :- source(semantic, research_papers, [
%       vector_store('papers.json'),
%       embedding_backend(python_onnx),  % Backend in Config
%       similarity_threshold(0.7),
%       top_k(10)
%   ]).
%
%   % Compile with target wrapper specified:
%   ?- compile(research_papers/2, [target(bash), output_dir('out/')], Code).
%
%   find_papers(Query, PaperId, Score) :-
%       research_papers(Query, PaperId, Score),
%       Score > 0.75.

:- use_module(library(option)).
:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

% Register this plugin at module load time
:- initialization(
    register_source_type(semantic, semantic_source),
    now
).

%% source_info(-Info) is det.
%
% Provides plugin metadata for the semantic source.
source_info(info(
    name('Semantic/Vector Search Source'),
    version('0.2.0'),
    description('Query documents via embedding vectors and cosine similarity'),
    supported_arities([2, 3]),  % (Query, Id), (Query, Id, Score)
    author('UnifyWeaver Team'),
    requires([
        'Embedding backends:',
        '  - python_onnx: Python 3.8+ with numpy, onnxruntime',
        '  - go_service: Go HTTP service',
        '  - rust_candle: Rust with Candle ML framework',
        '  - csharp_native: C# with LiteDB and ONNX Runtime',
        'Target wrappers:',
        '  - bash: Shell scripts',
        '  - powershell: PowerShell scripts',
        '  - csharp: C# code',
        '  - python: Python scripts'
    ]),
    options([
        'vector_store(Path) - Path to vector database (required)',
        'embedding_backend(Backend) - Backend type: python_onnx, go_service, rust_candle, csharp_native (required)',
        'embedding_backend(Backend, Config) - Backend with configuration',
        'similarity_threshold(Float) - Minimum similarity score 0.0-1.0 (default: 0.5)',
        'top_k(Int) - Maximum results to return (default: 10)',
        'similarity_metric(Atom) - Similarity function: cosine, euclidean, dot (default: cosine)',
        'normalize_vectors(Bool) - L2 normalization (default: true)',
        'cache_embeddings(Bool) - Cache query embeddings (default: true)',
        'cache_ttl(Seconds) - Cache time-to-live (default: 3600)'
    ])
)).

%% validate_config(+Config) is det.
%
% Validates semantic source configuration.
%
% @throws error if configuration is invalid
validate_config(Config) :-
    % Required: vector_store
    (   member(vector_store(StorePath), Config)
    ->  must_be(atom, StorePath)
    ;   throw(error(missing_option(vector_store),
                    'vector_store(Path) is required for semantic sources'))
    ),

    % Required: embedding_backend (will use default if not specified)
    (   member(embedding_backend(Backend), Config)
    ->  validate_backend_type(Backend)
    ;   member(embedding_backend(Backend, BackendConfig), Config)
    ->  validate_backend_type(Backend),
        validate_backend_config(Backend, BackendConfig)
    ;   true  % Will use default in augmentation phase
    ),

    % Optional: similarity_threshold
    (   member(similarity_threshold(Threshold), Config)
    ->  must_be(float, Threshold),
        (   Threshold >= 0.0, Threshold =< 1.0
        ->  true
        ;   throw(error(domain_error(float, Threshold),
                        'similarity_threshold must be between 0.0 and 1.0'))
        )
    ;   true
    ),

    % Optional: top_k
    (   member(top_k(K), Config)
    ->  must_be(integer, K),
        (   K > 0
        ->  true
        ;   throw(error(domain_error(positive_integer, K),
                        'top_k must be a positive integer'))
        )
    ;   true
    ),

    % Optional: similarity_metric
    (   member(similarity_metric(Metric), Config)
    ->  must_be(atom, Metric),
        (   member(Metric, [cosine, euclidean, dot])
        ->  true
        ;   throw(error(domain_error(similarity_metric, Metric),
                        'similarity_metric must be one of: cosine, euclidean, dot'))
        )
    ;   true
    ).

%% validate_backend_type(+Backend) is det.
%
% Validates embedding backend type.
validate_backend_type(Backend) :-
    must_be(atom, Backend),
    (   member(Backend, [python_onnx, go_service, rust_candle, csharp_native])
    ->  true
    ;   throw(error(domain_error(embedding_backend, Backend),
                    'Backend must be one of: python_onnx, go_service, rust_candle, csharp_native'))
    ).

%% validate_backend_config(+Backend, +Config) is det.
%
% Validates backend-specific configuration.

% Python ONNX backend
validate_backend_config(python_onnx, Config) :-
    !,
    (   atom(Config)
    ->  % Shorthand model name (e.g., 'all-MiniLM-L6-v2')
        true
    ;   is_list(Config)
    ->  % Full config with model_path, vocab_path, etc.
        (   member(model_path(_), Config) -> true
        ;   throw(error(missing_option(model_path),
                        'python_onnx backend requires model_path'))
        )
    ;   throw(error(type_error(backend_config, Config),
                    'python_onnx config must be atom or list'))
    ).

% Go service backend
validate_backend_config(go_service, Config) :-
    !,
    (   is_list(Config)
    ->  (   member(url(_), Config) -> true
        ;   throw(error(missing_option(url),
                        'go_service backend requires url'))
        )
    ;   throw(error(type_error(backend_config, Config),
                    'go_service config must be a list'))
    ).

% Rust Candle backend
validate_backend_config(rust_candle, Config) :-
    !,
    (   is_list(Config)
    ->  (   member(binary_path(_), Config) -> true
        ;   member(model_path(_), Config) -> true
        ;   throw(error(missing_option(binary_or_model),
                        'rust_candle backend requires binary_path or model_path'))
        )
    ;   throw(error(type_error(backend_config, Config),
                    'rust_candle config must be a list'))
    ).

% C# native backend
validate_backend_config(csharp_native, Config) :-
    !,
    (   atom(Config) ; is_list(Config)
    ->  true
    ;   throw(error(type_error(backend_config, Config),
                    'csharp_native config must be atom or list'))
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -GeneratedCode) is det.
%
% Generates code for semantic source based on target wrapper and embedding backend.
%
% Target (from Options): bash, powershell, csharp, python
% Backend (from Config): python_onnx, go_service, rust_candle, csharp_native
%
% @param Pred/Arity The predicate name and arity
% @param Config The source configuration options (includes embedding_backend)
% @param Options Compilation options (includes target)
% @param GeneratedCode The generated code (atom)
compile_source(Pred/Arity, Config, Options, GeneratedCode) :-
    format('  Compiling Semantic source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options for convenience
    append(Config, Options, AllOptions),

    % Extract target wrapper (from Options)
    option(target(Target), AllOptions, bash),

    % Extract embedding backend (from Config)
    (   option(embedding_backend(Backend, BackendConfig), AllOptions)
    ->  true
    ;   option(embedding_backend(Backend), AllOptions)
    ->  BackendConfig = default
    ;   Backend = python_onnx, BackendConfig = 'all-MiniLM-L6-v2'  % Default
    ),

    % Extract other configuration with defaults
    option(vector_store(StorePath), AllOptions),
    option(similarity_threshold(Threshold), AllOptions, 0.5),
    option(top_k(TopK), AllOptions, 10),
    option(similarity_metric(Metric), AllOptions, cosine),
    option(normalize_vectors(Normalize), AllOptions, true),

    % Validate Target × Backend combination is supported
    validate_combination(Target, Backend),

    % Dispatch to appropriate generator
    generate_wrapper_for_target_and_backend(
        Target, Backend, Pred, Arity,
        StorePath, BackendConfig, Threshold, TopK, Metric, Normalize,
        GeneratedCode
    ).

%% validate_combination(+Target, +Backend) is det.
%
% Validates that Target × Backend combination is supported.
validate_combination(Target, Backend) :-
    (   supported_combination(Target, Backend)
    ->  true
    ;   format(atom(Msg), 'Unsupported combination: target=~w with backend=~w', [Target, Backend]),
        throw(error(not_supported(Target, Backend), Msg))
    ).

%% supported_combination(?Target, ?Backend) is nondet.
%
% Defines supported Target × Backend combinations.

% Bash wrapper supports all backends
supported_combination(bash, python_onnx).
supported_combination(bash, go_service).
supported_combination(bash, rust_candle).
supported_combination(bash, csharp_native).  % Via mono/dotnet

% PowerShell wrapper supports all backends
supported_combination(powershell, python_onnx).
supported_combination(powershell, go_service).
supported_combination(powershell, rust_candle).
supported_combination(powershell, csharp_native).

% C# target only supports csharp_native backend natively
supported_combination(csharp, csharp_native).
% But could call external services
supported_combination(csharp, go_service).
supported_combination(csharp, rust_candle).  % Via Process.Start

% Python target supports Python backend natively
supported_combination(python, python_onnx).
% And can call services
supported_combination(python, go_service).
supported_combination(python, rust_candle).  % Via subprocess

%% generate_wrapper_for_target_and_backend(+Target, +Backend, ..., -Code) is det.
%
% Dispatches to specific generator based on Target × Backend combination.

% ===== BASH WRAPPERS =====

% Bash + Python ONNX
generate_wrapper_for_target_and_backend(
    bash, python_onnx, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, Normalize,
    BashCode
) :-
    !,
    % Expand backend config
    expand_python_onnx_config(BackendConfig, ExpandedConfig),
    option(model_path(ModelPath), ExpandedConfig, 'models/all-MiniLM-L6-v2.onnx'),
    option(vocab_path(VocabPath), ExpandedConfig, 'models/vocab.txt'),
    option(dimensions(_Dims), ExpandedConfig, 384),

    % Generate Python code for ONNX inference
    generate_python_onnx_code(
        ModelPath, VocabPath,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCode
    ),

    % Render bash template
    render_named_template(semantic_bash_python_onnx, [
        pred=Pred,
        python_code=PythonCode
    ], BashCode).

% Bash + Go Service
generate_wrapper_for_target_and_backend(
    bash, go_service, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, _Normalize,
    BashCode
) :-
    !,
    option(url(ServiceUrl), BackendConfig),
    render_named_template(semantic_bash_go_service, [
        pred=Pred,
        service_url=ServiceUrl,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], BashCode).

% Bash + Rust Candle
generate_wrapper_for_target_and_backend(
    bash, rust_candle, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, _Metric, _Normalize,
    BashCode
) :-
    !,
    (   option(binary_path(BinaryPath), BackendConfig)
    ->  true
    ;   BinaryPath = './rust_semantic_search'  % Default
    ),
    render_named_template(semantic_bash_rust_candle, [
        pred=Pred,
        binary_path=BinaryPath,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK
    ], BashCode).

% ===== FALLBACK =====

% Unsupported Target × Backend combinations
generate_wrapper_for_target_and_backend(Target, Backend, _, _, _, _, _, _, _, _, _) :-
    format(atom(Msg), 'Code generation not yet implemented for target=~w, backend=~w', [Target, Backend]),
    throw(error(not_implemented(Target, Backend), Msg)).

%% expand_python_onnx_config(+ConfigIn, -ConfigOut) is det.
%
% Expands simple model name to full configuration.
expand_python_onnx_config(ModelName, ExpandedConfig) :-
    atom(ModelName),
    !,
    onnx_model_defaults(ModelName, ExpandedConfig).

expand_python_onnx_config(default, ExpandedConfig) :-
    !,
    onnx_model_defaults('all-MiniLM-L6-v2', ExpandedConfig).

expand_python_onnx_config(Config, Config) :-
    is_list(Config),
    !.

expand_python_onnx_config(Config, _) :-
    throw(error(type_error(onnx_config, Config),
                'ONNX config must be atom (model name) or list (full config)')).

%% onnx_model_defaults(+ModelName, -Config) is det.
%
% Default configurations for common ONNX models.
onnx_model_defaults('all-MiniLM-L6-v2', [
    model_path('models/all-MiniLM-L6-v2.onnx'),
    vocab_path('models/vocab.txt'),
    dimensions(384),
    max_length(512)
]).

onnx_model_defaults('bge-small-en-v1.5', [
    model_path('models/bge-small-en-v1.5.onnx'),
    vocab_path('models/bge-vocab.txt'),
    dimensions(384),
    max_length(512)
]).

onnx_model_defaults('e5-small-v2', [
    model_path('models/e5-small-v2.onnx'),
    vocab_path('models/e5-vocab.txt'),
    dimensions(384),
    max_length(512)
]).

onnx_model_defaults(ModelName, _) :-
    format(atom(Msg), 'Unknown ONNX model: ~w. Please provide full config.', [ModelName]),
    throw(error(unknown_model(ModelName), Msg)).

%% generate_python_onnx_code(..., -PythonCode) is det.
%
% Generates Python code for ONNX embedding and vector search.
generate_python_onnx_code(ModelPath, VocabPath,
                          StorePath, Threshold, TopK, Metric, Normalize,
                          PythonCode) :-
    % Generate normalization code
    (   Normalize = true
    ->  NormCode = '    # L2 normalization\n    embedding = embedding / np.linalg.norm(embedding)'
    ;   NormCode = '    # No normalization'
    ),

    % Generate similarity function name
    similarity_function_name(Metric, SimFuncName),

    % Build complete Python code
    format(atom(PythonCode),
'import sys
import json
import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Compute negative euclidean distance (so higher is better)."""
    return -np.linalg.norm(a - b)

def dot_product(a, b):
    """Compute dot product similarity."""
    return np.dot(a, b)

def get_embedding_onnx(text):
    """Generate embedding using ONNX model."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Load model and tokenizer
        session = ort.InferenceSession("~w")
        tokenizer = AutoTokenizer.from_pretrained("~w")

        # Tokenize input
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)

        # Run inference
        outputs = session.run(None, {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        })

        # Extract embedding (mean pooling over sequence)
        embedding = outputs[0][0]
        attention_mask = inputs["attention_mask"][0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(embedding * mask_expanded, axis=0)
        sum_mask = np.clip(np.sum(attention_mask), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask

~w

        return embedding
    except Exception as e:
        print(f"Error generating embedding: {{e}}", file=sys.stderr)
        sys.exit(1)

def search_vectors(query_embedding, vector_store_path, top_k, threshold, similarity_func):
    """Search vector store for similar documents."""
    try:
        with open(vector_store_path, "r") as f:
            vector_store = json.load(f)

        results = []
        for doc_id, doc_vector in vector_store.items():
            doc_embedding = np.array(doc_vector)
            similarity = similarity_func(query_embedding, doc_embedding)

            if similarity >= threshold:
                results.append((doc_id, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    except FileNotFoundError:
        print(f"Error: Vector store not found: {{vector_store_path}}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in vector store: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error searching vectors: {{e}}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: <script> <query>", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    query_embedding = get_embedding_onnx(query)

    results = search_vectors(
        query_embedding,
        "~w",  # vector_store_path
        ~w,    # top_k
        ~w,    # threshold
        ~w     # similarity_func
    )

    for doc_id, score in results:
        print(f"{{doc_id}}:{{score:.4f}}")

if __name__ == "__main__":
    main()
', [ModelPath, VocabPath, NormCode, StorePath, TopK, Threshold, SimFuncName]).

%% similarity_function_name(+Metric, -FunctionName) is det.
%
% Maps similarity metric to Python function name.
similarity_function_name(cosine, cosine_similarity).
similarity_function_name(euclidean, euclidean_distance).
similarity_function_name(dot, dot_product).

%% ========== TEMPLATES ==========

:- multifile template_system:template/2.

% Bash + Python ONNX template
template_system:template(semantic_bash_python_onnx,
'#!/bin/bash
# {{pred}} - Semantic search (bash + Python ONNX backend)

{{pred}}() {
    local query="$1"

    if [ -z "$query" ]; then
        echo "Error: Query argument required" >&2
        return 1
    fi

    python3 - "$query" <<\'PYTHON_EOF\'
{{python_code}}
PYTHON_EOF
}

{{pred}}_stream() {
    {{pred}} "$@"
}

{{pred}}_batch() {
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            {{pred}} "$line"
        fi
    done
}

export -f {{pred}}
export -f {{pred}}_stream
export -f {{pred}}_batch
').

% Bash + Go Service template
template_system:template(semantic_bash_go_service,
'#!/bin/bash
# {{pred}} - Semantic search (bash + Go service backend)

{{pred}}() {
    local query="$1"

    if [ -z "$query" ]; then
        echo "Error: Query argument required" >&2
        return 1
    fi

    # Call Go service via HTTP
    curl -s -X POST "{{service_url}}/search" -H "Content-Type: application/json" -d "{\"query\": \"$query\", \"top_k\": {{top_k}}, \"threshold\": {{threshold}}, \"metric\": \"{{metric}}\", \"vector_store\": \"{{vector_store}}\"}" | jq -r \'.results[] | "\\(.id):\\(.score)"\'
}

{{pred}}_stream() {
    {{pred}} "$@"
}

export -f {{pred}}
export -f {{pred}}_stream
').

% Bash + Rust Candle template
template_system:template(semantic_bash_rust_candle,
'#!/bin/bash
# {{pred}} - Semantic search (bash + Rust Candle backend)

{{pred}}() {
    local query="$1"

    if [ -z "$query" ]; then
        echo "Error: Query argument required" >&2
        return 1
    fi

    # Call Rust binary
    "{{binary_path}}" search --query "$query" --vector-store "{{vector_store}}" --top-k {{top_k}} --threshold {{threshold}}
}

{{pred}}_stream() {
    {{pred}} "$@"
}

export -f {{pred}}
export -f {{pred}}_stream
').

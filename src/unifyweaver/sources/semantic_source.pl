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

% Allow non-contiguous clauses (generators are organized by target)
:- discontiguous generate_wrapper_for_target_and_backend/11.

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

% ===== PYTHON WRAPPERS =====

% Python + Python ONNX
generate_wrapper_for_target_and_backend(
    python, python_onnx, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, Normalize,
    PythonCode
) :-
    !,
    % Expand backend config
    expand_python_onnx_config(BackendConfig, ExpandedConfig),
    option(model_path(ModelPath), ExpandedConfig, 'models/all-MiniLM-L6-v2.onnx'),
    option(vocab_path(_VocabPath), ExpandedConfig, 'models/vocab.txt'),
    option(dimensions(_Dims), ExpandedConfig, 384),

    % For Python target, we use the model directory for tokenizer
    atom_concat(ModelDir, '/onnx/model.onnx', ModelPath),

    % Generate Python code directly (no wrapper needed)
    generate_python_onnx_code(
        ModelPath, ModelDir,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCodeInner
    ),

    % Wrap in a Python module with shebang
    render_named_template(semantic_python_python_onnx, [
        pred=Pred,
        python_code=PythonCodeInner,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], PythonCode).

% ===== POWERSHELL WRAPPERS =====

% PowerShell + Python ONNX
generate_wrapper_for_target_and_backend(
    powershell, python_onnx, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, Normalize,
    PowerShellCode
) :-
    !,
    % Expand backend config
    expand_python_onnx_config(BackendConfig, ExpandedConfig),
    option(model_path(ModelPath), ExpandedConfig, 'models/all-MiniLM-L6-v2.onnx'),
    option(vocab_path(_VocabPath), ExpandedConfig, 'models/vocab.txt'),
    option(dimensions(_Dims), ExpandedConfig, 384),

    % For PowerShell target, we use the model directory for tokenizer
    atom_concat(ModelDir, '/onnx/model.onnx', ModelPath),

    % Generate Python code (same as for Python target)
    generate_python_onnx_code(
        ModelPath, ModelDir,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCodeInner
    ),

    % Wrap in PowerShell with embedded Python
    render_named_template(semantic_powershell_python_onnx, [
        pred=Pred,
        python_code=PythonCodeInner,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], PowerShellCode).

% ===== C# IMPLEMENTATIONS =====

% C# + Native ONNX Runtime
generate_wrapper_for_target_and_backend(
    csharp, csharp_native, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, Normalize,
    CSharpCode
) :-
    !,
    % Expand backend config
    expand_csharp_native_config(BackendConfig, ExpandedConfig),
    option(model_path(ModelPath), ExpandedConfig, 'models/all-MiniLM-L6-v2.onnx'),
    option(vocab_path(VocabPath), ExpandedConfig, 'models/vocab.txt'),
    option(dimensions(Dims), ExpandedConfig, 384),

    % Generate similarity function name
    similarity_function_csharp(Metric, SimFuncName),

    % Generate normalization code
    (Normalize = true -> NormalizeFlag = 'true' ; NormalizeFlag = 'false'),

    % Render C# template
    render_named_template(semantic_csharp_native, [
        pred=Pred,
        model_path=ModelPath,
        vocab_path=VocabPath,
        dimensions=Dims,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric,
        similarity_func=SimFuncName,
        normalize=NormalizeFlag
    ], CSharpCode).

%% expand_csharp_native_config(+ConfigIn, -ConfigOut) is det.
%
% Expands simple model name to full configuration for C# native.
expand_csharp_native_config(ModelName, ExpandedConfig) :-
    atom(ModelName),
    !,
    csharp_native_model_defaults(ModelName, ExpandedConfig).

expand_csharp_native_config(default, ExpandedConfig) :-
    !,
    csharp_native_model_defaults('all-MiniLM-L6-v2', ExpandedConfig).

expand_csharp_native_config(Config, Config) :-
    is_list(Config),
    !.

expand_csharp_native_config(Config, _) :-
    throw(error(type_error(csharp_native_config, Config),
                'C# native config must be atom (model name) or list (full config)')).

%% csharp_native_model_defaults(+ModelName, -Config) is det.
%
% Default configurations for C# native ONNX models.
csharp_native_model_defaults('all-MiniLM-L6-v2', [
    model_path('models/all-MiniLM-L6-v2.onnx'),
    vocab_path('models/vocab.txt'),
    dimensions(384),
    max_length(512)
]).

%% similarity_function_csharp(+Metric, -FunctionName) is det.
%
% Maps similarity metric to C# method name.
similarity_function_csharp(cosine, 'CosineSimilarity').
similarity_function_csharp(euclidean, 'EuclideanDistance').
similarity_function_csharp(dot, 'DotProduct').

% ===== GO SERVICE WRAPPERS =====

% Python + Go Service
generate_wrapper_for_target_and_backend(
    python, go_service, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, _Normalize,
    PythonCode
) :-
    !,
    option(url(ServiceUrl), BackendConfig),
    render_named_template(semantic_python_go_service, [
        pred=Pred,
        service_url=ServiceUrl,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], PythonCode).

% PowerShell + Go Service
generate_wrapper_for_target_and_backend(
    powershell, go_service, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, _Normalize,
    PowerShellCode
) :-
    !,
    option(url(ServiceUrl), BackendConfig),
    render_named_template(semantic_powershell_go_service, [
        pred=Pred,
        service_url=ServiceUrl,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], PowerShellCode).

% C# + Go Service
generate_wrapper_for_target_and_backend(
    csharp, go_service, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, Metric, _Normalize,
    CSharpCode
) :-
    !,
    option(url(ServiceUrl), BackendConfig),
    render_named_template(semantic_csharp_go_service, [
        pred=Pred,
        service_url=ServiceUrl,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], CSharpCode).

% ===== RUST CANDLE WRAPPERS =====

% Python + Rust Candle
generate_wrapper_for_target_and_backend(
    python, rust_candle, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, _Metric, _Normalize,
    PythonCode
) :-
    !,
    (   option(binary_path(BinaryPath), BackendConfig)
    ->  true
    ;   BinaryPath = './rust_semantic_search'  % Default
    ),
    render_named_template(semantic_python_rust_candle, [
        pred=Pred,
        binary_path=BinaryPath,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK
    ], PythonCode).

% PowerShell + Rust Candle
generate_wrapper_for_target_and_backend(
    powershell, rust_candle, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, _Metric, _Normalize,
    PowerShellCode
) :-
    !,
    (   option(binary_path(BinaryPath), BackendConfig)
    ->  true
    ;   BinaryPath = './rust_semantic_search'  % Default
    ),
    render_named_template(semantic_powershell_rust_candle, [
        pred=Pred,
        binary_path=BinaryPath,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK
    ], PowerShellCode).

% C# + Rust Candle
generate_wrapper_for_target_and_backend(
    csharp, rust_candle, Pred, _Arity,
    StorePath, BackendConfig, Threshold, TopK, _Metric, _Normalize,
    CSharpCode
) :-
    !,
    (   option(binary_path(BinaryPath), BackendConfig)
    ->  true
    ;   BinaryPath = './rust_semantic_search'  % Default
    ),
    render_named_template(semantic_csharp_rust_candle, [
        pred=Pred,
        binary_path=BinaryPath,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK
    ], CSharpCode).

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

% Python + Python ONNX template
template_system:template(semantic_python_python_onnx,
'#!/usr/bin/env python3
"""
{{pred}} - Semantic search (Python + Python ONNX backend)
"""

{{python_code}}
').

% PowerShell + Python ONNX template
template_system:template(semantic_powershell_python_onnx,
'# {{pred}}.ps1 - Semantic search (PowerShell + Python ONNX backend)

function {{pred}} {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query
    )

    $pythonCode = @"
{{python_code}}
"@

    # Run Python code with query argument
    $pythonCode | python3 - $Query
}

function {{pred}}-Stream {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query
    )
    {{pred}} -Query $Query
}

function {{pred}}-Batch {
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string[]]$Lines
    )
    process {
        foreach ($line in $Lines) {
            if ($line) {
                {{pred}} -Query $line
            }
        }
    }
}

Export-ModuleMember -Function {{pred}}, {{pred}}-Stream, {{pred}}-Batch
').

% C# + Native ONNX Runtime template
template_system:template(semantic_csharp_native,
'// {{pred}}.cs - Semantic search (C# + native ONNX Runtime)
// Requires: Microsoft.ML.OnnxRuntime, System.Text.Json

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SemanticSearch
{
    public class {{pred}}
    {
        private readonly InferenceSession _session;
        private readonly string _vectorStorePath;
        private readonly float _threshold;
        private readonly int _topK;
        private readonly Dictionary<string, float[]> _vectorStore;

        public {{pred}}()
        {
            _session = new InferenceSession("{{model_path}}");
            _vectorStorePath = "{{vector_store}}";
            _threshold = {{threshold}}f;
            _topK = {{top_k}};

            // Load vector store
            var json = File.ReadAllText(_vectorStorePath);
            var rawStore = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            _vectorStore = new Dictionary<string, float[]>();

            foreach (var kvp in rawStore)
            {
                var embedding = kvp.Value.GetProperty("embedding")
                    .EnumerateArray()
                    .Select(e => (float)e.GetDouble())
                    .ToArray();
                _vectorStore[kvp.Key] = embedding;
            }
        }

        public List<(string Id, float Score)> Search(string query)
        {
            var queryEmbedding = GetEmbedding(query);
            var results = new List<(string, float)>();

            foreach (var kvp in _vectorStore)
            {
                var similarity = {{similarity_func}}(queryEmbedding, kvp.Value);
                if (similarity >= _threshold)
                {
                    results.Add((kvp.Key, similarity));
                }
            }

            results.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            return results.Take(_topK).ToList();
        }

        private float[] GetEmbedding(string text)
        {
            // Simple tokenization (word-based, for demo)
            var tokens = text.Split(\' \');
            var inputIds = new long[tokens.Length];
            var attentionMask = new long[tokens.Length];

            for (int i = 0; i < tokens.Length; i++)
            {
                inputIds[i] = tokens[i].GetHashCode() % 30000;
                attentionMask[i] = 1;
            }

            var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, tokens.Length });
            var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, tokens.Length });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            // Mean pooling and normalization
            var pooled = MeanPool(output, attentionMask);
            if ({{normalize}})
            {
                Normalize(pooled);
            }

            return pooled;
        }

        private static float[] MeanPool(float[] embeddings, long[] mask)
        {
            var dims = {{dimensions}};
            var pooled = new float[dims];
            var sumMask = mask.Sum();

            for (int i = 0; i < dims; i++)
            {
                pooled[i] = embeddings[i] / sumMask;
            }

            return pooled;
        }

        private static void Normalize(float[] vec)
        {
            var norm = (float)Math.Sqrt(vec.Sum(x => x * x));
            if (norm > 0)
            {
                for (int i = 0; i < vec.Length; i++)
                {
                    vec[i] /= norm;
                }
            }
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            var dot = a.Zip(b, (x, y) => x * y).Sum();
            var normA = Math.Sqrt(a.Sum(x => x * x));
            var normB = Math.Sqrt(b.Sum(x => x * x));
            return (float)(dot / (normA * normB));
        }

        private static float EuclideanDistance(float[] a, float[] b)
        {
            return -(float)Math.Sqrt(a.Zip(b, (x, y) => (x - y) * (x - y)).Sum());
        }

        private static float DotProduct(float[] a, float[] b)
        {
            return a.Zip(b, (x, y) => x * y).Sum();
        }

        public static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("Usage: {{pred}} <query>");
                Environment.Exit(1);
            }

            var searcher = new {{pred}}();
            var results = searcher.Search(args[0]);

            foreach (var (id, score) in results)
            {
                Console.WriteLine($"{id}:{score:F4}");
            }
        }
    }
}
').

% Python + Go Service template
template_system:template(semantic_python_go_service,
'#!/usr/bin/env python3
"""
{{pred}} - Semantic search (Python + Go service backend)
"""

import sys
import json
import urllib.request
import urllib.parse

def {{pred}}(query, top_k={{top_k}}, threshold={{threshold}}):
    """Search using Go service backend."""
    url = "{{service_url}}/search"
    data = {
        "query": query,
        "top_k": top_k,
        "threshold": threshold,
        "metric": "{{metric}}",
        "vector_store": "{{vector_store}}"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("results", [])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {{pred}} <query>", file=sys.stderr)
        sys.exit(1)

    results = {{pred}}(sys.argv[1])
    for result in results:
        print(f"{result[\"id\"]}:{result[\"score\"]:.4f}")
').

% PowerShell + Go Service template
template_system:template(semantic_powershell_go_service,
'# {{pred}}.ps1 - Semantic search (PowerShell + Go service backend)

function {{pred}} {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [int]$TopK = {{top_k}},
        [float]$Threshold = {{threshold}}
    )

    $url = "{{service_url}}/search"
    $body = @{
        query = $Query
        top_k = $TopK
        threshold = $Threshold
        metric = "{{metric}}"
        vector_store = "{{vector_store}}"
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"
        foreach ($result in $response.results) {
            Write-Output "$($result.id):$($result.score.ToString(\"F4\"))"
        }
    }
    catch {
        Write-Error "Error calling Go service: $_"
        exit 1
    }
}

function {{pred}}-Stream {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query
    )
    {{pred}} -Query $Query
}

function {{pred}}-Batch {
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string[]]$Lines
    )
    process {
        foreach ($line in $Lines) {
            if ($line) {
                {{pred}} -Query $line
            }
        }
    }
}

Export-ModuleMember -Function {{pred}}, {{pred}}-Stream, {{pred}}-Batch
').

% C# + Go Service template
template_system:template(semantic_csharp_go_service,
'// {{pred}}.cs - Semantic search (C# + Go service backend)

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace SemanticSearch
{
    public class {{pred}}
    {
        private readonly HttpClient _httpClient;
        private readonly string _serviceUrl;
        private readonly string _vectorStore;
        private readonly float _threshold;
        private readonly int _topK;
        private readonly string _metric;

        public {{pred}}()
        {
            _httpClient = new HttpClient();
            _serviceUrl = "{{service_url}}";
            _vectorStore = "{{vector_store}}";
            _threshold = {{threshold}}f;
            _topK = {{top_k}};
            _metric = "{{metric}}";
        }

        public async Task<List<SearchResult>> SearchAsync(string query)
        {
            var request = new SearchRequest
            {
                Query = query,
                TopK = _topK,
                Threshold = _threshold,
                Metric = _metric,
                VectorStore = _vectorStore
            };

            var json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            try
            {
                var response = await _httpClient.PostAsync($"{_serviceUrl}/search", content);
                response.EnsureSuccessStatusCode();

                var responseJson = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<SearchResponse>(responseJson);

                return result?.Results ?? new List<SearchResult>();
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Error: {e.Message}");
                Environment.Exit(1);
                return null;
            }
        }

        public static async Task Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("Usage: {{pred}} <query>");
                Environment.Exit(1);
            }

            var searcher = new {{pred}}();
            var results = await searcher.SearchAsync(args[0]);

            foreach (var result in results)
            {
                Console.WriteLine($"{result.Id}:{result.Score:F4}");
            }
        }
    }

    public class SearchRequest
    {
        public string Query { get; set; }
        public int TopK { get; set; }
        public float Threshold { get; set; }
        public string Metric { get; set; }
        public string VectorStore { get; set; }
    }

    public class SearchResponse
    {
        public List<SearchResult> Results { get; set; }
    }

    public class SearchResult
    {
        public string Id { get; set; }
        public float Score { get; set; }
    }
}
').

% Python + Rust Candle template
template_system:template(semantic_python_rust_candle,
'#!/usr/bin/env python3
"""
{{pred}} - Semantic search (Python + Rust Candle backend)
"""

import sys
import subprocess
import json

def {{pred}}(query, top_k={{top_k}}, threshold={{threshold}}):
    """Search using Rust Candle backend."""
    try:
        result = subprocess.run(
            [
                "{{binary_path}}",
                "search",
                "--query", query,
                "--vector-store", "{{vector_store}}",
                "--top-k", str(top_k),
                "--threshold", str(threshold)
            ],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse output (format: doc_id:score)
        results = []
        for line in result.stdout.strip().split("\\n"):
            if ":" in line:
                doc_id, score = line.split(":", 1)
                results.append({"id": doc_id, "score": float(score)})

        return results
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {{pred}} <query>", file=sys.stderr)
        sys.exit(1)

    results = {{pred}}(sys.argv[1])
    for result in results:
        print(f"{result[\"id\"]}:{result[\"score\"]:.4f}")
').

% PowerShell + Rust Candle template
template_system:template(semantic_powershell_rust_candle,
'# {{pred}}.ps1 - Semantic search (PowerShell + Rust Candle backend)

function {{pred}} {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query,
        [int]$TopK = {{top_k}},
        [float]$Threshold = {{threshold}}
    )

    try {
        $result = & "{{binary_path}}" search `
            --query $Query `
            --vector-store "{{vector_store}}" `
            --top-k $TopK `
            --threshold $Threshold

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Rust binary failed with exit code $LASTEXITCODE"
            exit 1
        }

        $result
    }
    catch {
        Write-Error "Error calling Rust binary: $_"
        exit 1
    }
}

function {{pred}}-Stream {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Query
    )
    {{pred}} -Query $Query
}

function {{pred}}-Batch {
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string[]]$Lines
    )
    process {
        foreach ($line in $Lines) {
            if ($line) {
                {{pred}} -Query $line
            }
        }
    }
}

Export-ModuleMember -Function {{pred}}, {{pred}}-Stream, {{pred}}-Batch
').

% C# + Rust Candle template
template_system:template(semantic_csharp_rust_candle,
'// {{pred}}.cs - Semantic search (C# + Rust Candle backend)

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SemanticSearch
{
    public class {{pred}}
    {
        private readonly string _binaryPath;
        private readonly string _vectorStore;
        private readonly float _threshold;
        private readonly int _topK;

        public {{pred}}()
        {
            _binaryPath = "{{binary_path}}";
            _vectorStore = "{{vector_store}}";
            _threshold = {{threshold}}f;
            _topK = {{top_k}};
        }

        public List<SearchResult> Search(string query)
        {
            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = _binaryPath,
                    Arguments = $"search --query \"{query}\" --vector-store \"{_vectorStore}\" --top-k {_topK} --threshold {_threshold}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(startInfo);
                var output = process.StandardOutput.ReadToEnd();
                var error = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    Console.Error.WriteLine($"Error: {error}");
                    Environment.Exit(1);
                }

                // Parse output (format: doc_id:score)
                return output.Split(new[] { \'\\n\' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(line =>
                    {
                        var parts = line.Split(\':\');
                        return new SearchResult
                        {
                            Id = parts[0],
                            Score = float.Parse(parts[1])
                        };
                    })
                    .ToList();
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Error: {e.Message}");
                Environment.Exit(1);
                return null;
            }
        }

        public static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("Usage: {{pred}} <query>");
                Environment.Exit(1);
            }

            var searcher = new {{pred}}();
            var results = searcher.Search(args[0]);

            foreach (var result in results)
            {
                Console.WriteLine($"{result.Id}:{result.Score:F4}");
            }
        }
    }

    public class SearchResult
    {
        public string Id { get; set; }
        public float Score { get; set; }
    }
}
').

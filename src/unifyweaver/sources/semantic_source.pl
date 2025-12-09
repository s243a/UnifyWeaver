:- module(semantic_source, []).

%% Semantic/Vector Search Source Plugin
%
% Provides semantic search capabilities via embedding vectors and similarity search.
%
% This plugin enables semantic retrieval over document collections using:
% - ONNX embedding models (local inference)
% - External embedding services (Ollama, Together AI, OpenAI)
% - Rust Candle embeddings (GPU-accelerated)
%
% Example usage:
%   :- source(semantic, research_papers, [
%       vector_store('papers.litedb'),
%       embedding_model(onnx, 'all-MiniLM-L6-v2'),
%       similarity_threshold(0.7),
%       top_k(10)
%   ]).
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
    version('0.1.0'),
    description('Query documents via embedding vectors and cosine similarity'),
    supported_arities([2, 3]),  % (Query, Id), (Query, Id, Score)
    author('UnifyWeaver Team'),
    requires([
        'Python 3.8+ with numpy, onnxruntime (for ONNX embeddings)',
        'or C# with LiteDB and ONNX Runtime (for C# target)',
        'or Rust with Candle ML framework (for Rust target)'
    ]),
    options([
        'vector_store(Path) - Path to vector database (required)',
        'embedding_model(Type, Config) - Embedding provider configuration (required)',
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

    % Required: embedding_model (will use default if not specified)
    (   member(embedding_model(Type, ModelConfig), Config)
    ->  validate_embedding_model_type(Type, ModelConfig)
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

%% validate_embedding_model_type(+Type, +Config) is det.
%
% Validates specific embedding model configuration.
validate_embedding_model_type(onnx, Config) :-
    !,
    % For ONNX, we can accept either:
    % 1. Simple atom: onnx('all-MiniLM-L6-v2')
    % 2. Full config: onnx([model_path(...), vocab_path(...), dimensions(...)])
    (   atom(Config)
    ->  true  % Simple model name, will be expanded
    ;   is_list(Config)
    ->  % Validate list config
        (   member(model_path(_), Config) -> true
        ;   throw(error(missing_option(model_path),
                        'ONNX embedding model requires model_path in config'))
        )
    ;   throw(error(type_error(embedding_config, Config),
                    'ONNX config must be atom or list'))
    ).

validate_embedding_model_type(service, Config) :-
    !,
    (   is_list(Config)
    ->  (   member(endpoint(_), Config) -> true
        ;   throw(error(missing_option(endpoint),
                        'Service embedding model requires endpoint in config'))
        )
    ;   throw(error(type_error(embedding_config, Config),
                    'Service config must be a list'))
    ).

validate_embedding_model_type(rust_candle, Config) :-
    !,
    (   is_list(Config)
    ->  (   member(model_path(_), Config) -> true
        ;   throw(error(missing_option(model_path),
                        'Rust Candle embedding model requires model_path in config'))
        )
    ;   throw(error(type_error(embedding_config, Config),
                    'Rust Candle config must be a list'))
    ).

validate_embedding_model_type(Type, _Config) :-
    throw(error(domain_error(embedding_model_type, Type),
                'Unknown embedding model type. Must be one of: onnx, service, rust_candle')).

%% compile_source(+Pred/Arity, +Config, +Options, -GeneratedCode) is det.
%
% Generates code for semantic source based on target platform.
%
% @param Pred/Arity The predicate name and arity
% @param Config The source configuration options
% @param Options Compilation options (target, output_dir, etc.)
% @param GeneratedCode The generated code (atom)
compile_source(Pred/Arity, Config, Options, GeneratedCode) :-
    format('  Compiling Semantic source: ~w/~w~n', [Pred, Arity]),

    % Extract configuration with defaults
    option(vector_store(StorePath), Config),
    option(embedding_model(EmbedType, EmbedConfig), Config, onnx('all-MiniLM-L6-v2')),
    option(similarity_threshold(Threshold), Config, 0.5),
    option(top_k(TopK), Config, 10),
    option(similarity_metric(Metric), Config, cosine),
    option(normalize_vectors(Normalize), Config, true),

    % Determine target language (default: bash)
    option(target(Target), Options, bash),

    % Generate code based on target and embedding type
    generate_code(Target, EmbedType, Pred, Arity,
                  StorePath, EmbedConfig, Threshold, TopK, Metric, Normalize,
                  GeneratedCode).

%% generate_code(+Target, +EmbedType, +Pred, +Arity, +StorePath, +EmbedConfig,
%%                +Threshold, +TopK, +Metric, +Normalize, -Code) is det.
%
% Target-specific code generation dispatcher.

% Bash target with ONNX embeddings
generate_code(bash, onnx, Pred, _Arity,
              StorePath, EmbedConfig, Threshold, TopK, Metric, Normalize,
              BashCode) :-
    !,
    % Expand model config if it's just a model name
    expand_onnx_config(EmbedConfig, ExpandedConfig),

    % Extract ONNX-specific configuration
    option(model_path(ModelPath), ExpandedConfig, 'models/all-MiniLM-L6-v2.onnx'),
    option(vocab_path(VocabPath), ExpandedConfig, 'models/vocab.txt'),
    option(dimensions(Dims), ExpandedConfig, 384),

    % Generate Python code for ONNX inference
    generate_onnx_python_code(
        ModelPath, VocabPath, Dims,
        StorePath, Threshold, TopK, Metric, Normalize,
        PythonCode
    ),

    % Render bash template with embedded Python
    render_named_template(semantic_bash_onnx, [
        pred=Pred,
        python_code=PythonCode,
        vector_store=StorePath,
        threshold=Threshold,
        top_k=TopK,
        metric=Metric
    ], BashCode).

% Fallback for unsupported target/embedding combinations
generate_code(Target, EmbedType, _Pred, _Arity, _, _, _, _, _, _, _) :-
    format(atom(Msg), 'Unsupported combination: target=~w, embedding_model=~w', [Target, EmbedType]),
    throw(error(not_implemented(Target, EmbedType), Msg)).

%% expand_onnx_config(+ConfigIn, -ConfigOut) is det.
%
% Expands simple model name to full configuration.
expand_onnx_config(ModelName, ExpandedConfig) :-
    atom(ModelName),
    !,
    % Map common model names to their configurations
    onnx_model_defaults(ModelName, ExpandedConfig).

expand_onnx_config(Config, Config) :-
    is_list(Config),
    !.

expand_onnx_config(Config, _) :-
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

%% generate_onnx_python_code(..., -PythonCode) is det.
%
% Generates Python code for ONNX embedding and vector search.
generate_onnx_python_code(ModelPath, VocabPath, _Dims,
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

        # Extract embedding (typically from [CLS] token or mean pooling)
        embedding = outputs[0][0]  # Shape: (seq_len, hidden_dim)

        # Mean pooling over sequence length
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
        # Load vector store
        with open(vector_store_path, "r") as f:
            vector_store = json.load(f)

        results = []
        for doc_id, doc_vector in vector_store.items():
            doc_embedding = np.array(doc_vector)
            similarity = similarity_func(query_embedding, doc_embedding)

            if similarity >= threshold:
                results.append((doc_id, float(similarity)))

        # Sort by similarity (descending) and take top k
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

    # Generate query embedding
    query_embedding = get_embedding_onnx(query)

    # Search vector store
    results = search_vectors(
        query_embedding,
        "~w",  # vector_store_path
        ~w,    # top_k
        ~w,    # threshold
        ~w     # similarity_func
    )

    # Output results (format: doc_id:score)
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

%% Bash template with embedded Python ONNX code
:- multifile template_system:template/2.

template_system:template(semantic_bash_onnx,
'#!/bin/bash
# {{pred}} - Semantic search source (ONNX embeddings)
# Vector store: {{vector_store}}
# Similarity threshold: {{threshold}}
# Top K: {{top_k}}
# Metric: {{metric}}

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

# Stream interface (for pipeline integration)
{{pred}}_stream() {
    {{pred}} "$@"
}

# Batch query interface
{{pred}}_batch() {
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            {{pred}} "$line"
        fi
    done
}

# Export for subshell access
export -f {{pred}}
export -f {{pred}}_stream
export -f {{pred}}_batch
').

/**
 * ML Backend Abstraction Layer
 *
 * Provides a unified interface for browser-based ML libraries.
 * Supports: Transformers.js, ONNX Runtime Web, TensorFlow.js
 *
 * Usage:
 *   const backend = await MLBackend.create('transformers.js');
 *   const embeddings = await backend.embed(['text1', 'text2']);
 *   const compressed = await backend.compress(W_matrix, rank);
 */

class MLBackendBase {
    constructor(name) {
        this.name = name;
        this.ready = false;
        this.model = null;
    }

    async initialize(modelId) {
        throw new Error('Not implemented');
    }

    async embed(texts, options = {}) {
        throw new Error('Not implemented');
    }

    async compress(matrix, rank) {
        // Default: SVD-based compression (works for all backends)
        // Can be overridden for learned compression
        const { U, S, V } = this._svd(matrix);
        const k = Math.min(rank, S.length);
        return this._lowRankApprox(U, S, V, k);
    }

    // Simple SVD using power iteration (fallback)
    _svd(matrix) {
        // For proper SVD, use Pyodide's NumPy
        // This is a placeholder - real implementation uses backend-specific methods
        throw new Error('SVD requires Pyodide or backend-specific implementation');
    }

    getStatus() {
        return {
            name: this.name,
            ready: this.ready,
            model: this.model
        };
    }

    dispose() {
        this.ready = false;
        this.model = null;
    }
}


/**
 * Transformers.js Backend
 * Uses @xenova/transformers for HuggingFace models in browser
 */
class TransformersJSBackend extends MLBackendBase {
    constructor() {
        super('Transformers.js');
        this.pipeline = null;
        this.extractor = null;
    }

    async initialize(modelId = 'Xenova/all-MiniLM-L6-v2') {
        if (this.ready && this.model === modelId) {
            return; // Already initialized with this model
        }

        console.log(`[ML Backend] Loading Transformers.js with model: ${modelId}`);

        // Dynamic import
        const { pipeline, env } = await import(
            'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0'
        );

        // Configure for browser
        env.allowLocalModels = false;
        env.useBrowserCache = true;

        this.pipeline = pipeline;
        this.extractor = await pipeline('feature-extraction', modelId, {
            quantized: true // Use quantized model for faster loading
        });

        this.model = modelId;
        this.ready = true;
        console.log(`[ML Backend] Transformers.js ready`);
    }

    async embed(texts, options = {}) {
        if (!this.ready) {
            throw new Error('Backend not initialized. Call initialize() first.');
        }

        const {
            pooling = 'mean',
            normalize = true,
            batchSize = 8
        } = options;

        // Process in batches
        const results = [];
        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            const output = await this.extractor(batch, { pooling, normalize });

            // Convert to array format
            for (let j = 0; j < batch.length; j++) {
                results.push(Array.from(output[j].data));
            }
        }

        return results;
    }

    dispose() {
        super.dispose();
        this.extractor = null;
        this.pipeline = null;
    }
}


/**
 * ONNX Runtime Web Backend
 * Uses onnxruntime-web for running ONNX models
 */
class ONNXBackend extends MLBackendBase {
    constructor() {
        super('ONNX Runtime Web');
        this.session = null;
        this.tokenizer = null;
    }

    async initialize(modelPath) {
        console.log(`[ML Backend] Loading ONNX Runtime Web with model: ${modelPath}`);

        const ort = await import(
            'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js'
        );

        // Load ONNX model
        this.session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu', 'wasm'] // Try WebGPU first, fallback to WASM
        });

        this.model = modelPath;
        this.ready = true;
        console.log(`[ML Backend] ONNX Runtime ready`);
    }

    async embed(texts, options = {}) {
        if (!this.ready) {
            throw new Error('Backend not initialized. Call initialize() first.');
        }

        // Note: ONNX requires tokenization - would need to include tokenizer
        // This is a placeholder - real implementation needs model-specific tokenizer
        throw new Error('ONNX embed() requires tokenizer implementation for specific model');
    }

    dispose() {
        super.dispose();
        if (this.session) {
            this.session.release();
            this.session = null;
        }
    }
}


/**
 * TensorFlow.js Backend
 * Uses tfjs for TensorFlow models in browser
 */
class TFJSBackend extends MLBackendBase {
    constructor() {
        super('TensorFlow.js');
        this.tf = null;
        this.useModel = null;
    }

    async initialize(modelType = 'universal-sentence-encoder') {
        console.log(`[ML Backend] Loading TensorFlow.js with model: ${modelType}`);

        // Load TensorFlow.js
        this.tf = await import(
            'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0'
        );

        // Try to use WebGPU if available
        if (navigator.gpu) {
            await this.tf.setBackend('webgpu');
        } else {
            await this.tf.setBackend('webgl');
        }

        // Load Universal Sentence Encoder
        if (modelType === 'universal-sentence-encoder') {
            const use = await import(
                'https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder@1.3.3'
            );
            this.useModel = await use.load();
        }

        this.model = modelType;
        this.ready = true;
        console.log(`[ML Backend] TensorFlow.js ready with ${this.tf.getBackend()} backend`);
    }

    async embed(texts, options = {}) {
        if (!this.ready || !this.useModel) {
            throw new Error('Backend not initialized. Call initialize() first.');
        }

        const embeddings = await this.useModel.embed(texts);
        const result = await embeddings.array();
        embeddings.dispose();

        return result;
    }

    dispose() {
        super.dispose();
        if (this.useModel) {
            this.useModel = null;
        }
        this.tf = null;
    }
}


/**
 * Pyodide Backend (NumPy/SciPy)
 * Uses existing Pyodide instance for numerical operations
 */
class PyodideBackend extends MLBackendBase {
    constructor(pyodideInstance) {
        super('Pyodide (NumPy)');
        this.pyodide = pyodideInstance;
    }

    async initialize() {
        if (!this.pyodide) {
            throw new Error('Pyodide instance not provided');
        }
        this.ready = true;
        this.model = 'numpy';
        console.log(`[ML Backend] Pyodide backend ready`);
    }

    async embed(texts, options = {}) {
        // Pyodide doesn't do embeddings directly - use for matrix ops
        throw new Error('Pyodide backend does not support embedding. Use for matrix operations.');
    }

    async compress(matrix, rank) {
        if (!this.ready) {
            throw new Error('Backend not initialized');
        }

        // Use NumPy SVD for compression
        const matrixJson = JSON.stringify(matrix);
        const result = await this.pyodide.runPythonAsync(`
import numpy as np
import json

matrix = np.array(json.loads('${matrixJson}'))
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
k = min(${rank}, len(S))

# Low-rank approximation
W_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Also return compression stats
compression_ratio = (matrix.size) / (U[:, :k].size + k + Vt[:k, :].size)
reconstruction_error = np.linalg.norm(matrix - W_compressed, 'fro') / np.linalg.norm(matrix, 'fro')

json.dumps({
    'compressed': W_compressed.tolist(),
    'rank': k,
    'compression_ratio': float(compression_ratio),
    'reconstruction_error': float(reconstruction_error),
    'singular_values': S[:k].tolist()
})
        `);

        return JSON.parse(result);
    }

    async matmul(A, B) {
        const result = await this.pyodide.runPythonAsync(`
import numpy as np
import json
A = np.array(${JSON.stringify(A)})
B = np.array(${JSON.stringify(B)})
json.dumps((A @ B).tolist())
        `);
        return JSON.parse(result);
    }
}


/**
 * ML Backend Factory
 */
const MLBackend = {
    backends: {
        'transformers.js': TransformersJSBackend,
        'onnx': ONNXBackend,
        'tfjs': TFJSBackend,
        'pyodide': PyodideBackend
    },

    defaultModels: {
        'transformers.js': 'Xenova/all-MiniLM-L6-v2',
        'onnx': null, // Requires user to specify model path
        'tfjs': 'universal-sentence-encoder',
        'pyodide': 'numpy'
    },

    /**
     * Create and initialize a backend
     * @param {string} type - Backend type: 'transformers.js', 'onnx', 'tfjs', 'pyodide'
     * @param {object} options - Backend-specific options
     * @returns {MLBackendBase} Initialized backend
     */
    async create(type, options = {}) {
        const BackendClass = this.backends[type];
        if (!BackendClass) {
            throw new Error(`Unknown backend type: ${type}. Available: ${Object.keys(this.backends).join(', ')}`);
        }

        let backend;
        if (type === 'pyodide') {
            backend = new BackendClass(options.pyodide);
        } else {
            backend = new BackendClass();
        }

        const model = options.model || this.defaultModels[type];
        await backend.initialize(model);

        return backend;
    },

    /**
     * List available backends
     */
    list() {
        return Object.keys(this.backends).map(key => ({
            id: key,
            name: this.backends[key].name || key,
            defaultModel: this.defaultModels[key]
        }));
    }
};


// Export for ES modules and global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MLBackend, MLBackendBase };
}
if (typeof window !== 'undefined') {
    window.MLBackend = MLBackend;
}

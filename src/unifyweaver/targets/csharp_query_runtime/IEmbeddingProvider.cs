// SPDX-License-Identifier: MIT OR Apache-2.0
namespace UnifyWeaver.QueryRuntime
{
    /// <summary>
    /// Interface for pluggable embedding providers.
    /// Implementations can use ONNX, API services (Together AI), or local models (Ollama).
    /// </summary>
    public interface IEmbeddingProvider
    {
        /// <summary>
        /// Generate an embedding vector for the given text.
        /// </summary>
        /// <param name="text">Input text to embed</param>
        /// <returns>Embedding vector as array of doubles</returns>
        double[] GetEmbedding(string text);

        /// <summary>
        /// The dimensionality of the embedding vectors produced by this provider.
        /// </summary>
        int Dimensions { get; }

        /// <summary>
        /// The name of the embedding model (e.g., "all-MiniLM-L6-v2").
        /// </summary>
        string ModelName { get; }
    }
}

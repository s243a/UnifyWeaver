// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace UnifyWeaver.QueryRuntime
{
    /// <summary>
    /// ONNX-based embedding provider for sentence-transformers models.
    /// Supports all-MiniLM-L6-v2 and similar BERT-based models.
    /// </summary>
    public sealed class OnnxEmbeddingProvider : IEmbeddingProvider, IDisposable
    {
        private readonly InferenceSession _session;
        private readonly Dictionary<string, int> _vocab;
        private readonly int _maxLength;
        private const int ClsTokenId = 101;  // [CLS]
        private const int SepTokenId = 102;  // [SEP]
        private const int PadTokenId = 0;    // [PAD]
        private const int UnkTokenId = 100;  // [UNK]

        public int Dimensions { get; }
        public string ModelName { get; }

        /// <summary>
        /// Creates an ONNX embedding provider.
        /// </summary>
        /// <param name="modelPath">Path to the .onnx model file</param>
        /// <param name="vocabPath">Path to the vocab.txt file</param>
        /// <param name="dimensions">Embedding dimensions (384 for all-MiniLM-L6-v2)</param>
        /// <param name="maxLength">Maximum sequence length in tokens (default 512)</param>
        /// <param name="modelName">Display name for the model</param>
        public OnnxEmbeddingProvider(
            string modelPath,
            string vocabPath,
            int dimensions = 384,
            int maxLength = 512,
            string modelName = "all-MiniLM-L6-v2")
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"ONNX model not found: {modelPath}");
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException($"Vocab not found: {vocabPath}");

            _session = new InferenceSession(modelPath);

            // Load vocabulary
            _vocab = new Dictionary<string, int>();
            var lines = File.ReadAllLines(vocabPath);
            for (int i = 0; i < lines.Length; i++)
            {
                _vocab[lines[i]] = i;
            }

            _maxLength = maxLength;
            Dimensions = dimensions;
            ModelName = modelName;
        }

        public double[] GetEmbedding(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new double[Dimensions];
            }

            // Simple tokenization (basic word-level + lowercase)
            var tokens = SimpleTokenize(text);

            // Convert to IDs with [CLS] and [SEP]
            var tokenIds = new List<int> { ClsTokenId };
            foreach (var token in tokens)
            {
                if (_vocab.TryGetValue(token, out var id))
                {
                    tokenIds.Add(id);
                }
                else
                {
                    tokenIds.Add(UnkTokenId);
                }

                if (tokenIds.Count >= _maxLength - 1)  // Leave room for [SEP]
                    break;
            }
            tokenIds.Add(SepTokenId);

            var inputIdsArray = tokenIds.ToArray();
            var attentionMask = Enumerable.Repeat(1, inputIdsArray.Length).ToArray();

            // Convert to int64 for ONNX
            var inputIdsLong = inputIdsArray.Select(t => (long)t).ToArray();
            var attentionMaskLong = attentionMask.Select(m => (long)m).ToArray();

            // Create token type IDs (all zeros for single sequence)
            var tokenTypeIds = new long[inputIdsLong.Length];

            // Create tensors with batch dimension
            var inputIdsTensor = new DenseTensor<long>(inputIdsLong, new[] { 1, inputIdsLong.Length });
            var attentionMaskTensor = new DenseTensor<long>(attentionMaskLong, new[] { 1, attentionMaskLong.Length });
            var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, tokenTypeIds.Length });

            // Create input dictionary
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
            };

            // Run inference
            using var results = _session.Run(inputs);

            // Get the output tensor (last_hidden_state or token_embeddings)
            // Shape: [batch_size, sequence_length, hidden_size]
            var output = results.First().AsTensor<float>();

            // Mean pooling: average across sequence length, weighted by attention mask
            var embedding = MeanPooling(output, attentionMaskLong);

            // Normalize the embedding (L2 normalization)
            return Normalize(embedding);
        }

        private double[] MeanPooling(Tensor<float> hiddenStates, long[] attentionMask)
        {
            // hiddenStates shape: [1, seq_len, hidden_dim]
            var seqLen = hiddenStates.Dimensions[1];
            var hiddenDim = hiddenStates.Dimensions[2];

            var pooled = new double[hiddenDim];
            var maskSum = attentionMask.Sum();

            if (maskSum == 0)
                return pooled;

            for (int d = 0; d < hiddenDim; d++)
            {
                double sum = 0;
                for (int s = 0; s < seqLen; s++)
                {
                    if (attentionMask[s] > 0)
                    {
                        sum += hiddenStates[0, s, d];
                    }
                }
                pooled[d] = sum / maskSum;
            }

            return pooled;
        }

        private double[] Normalize(double[] vector)
        {
            var norm = Math.Sqrt(vector.Sum(v => v * v));
            if (norm == 0)
                return vector;

            return vector.Select(v => v / norm).ToArray();
        }

        private List<string> SimpleTokenize(string text)
        {
            // Basic tokenization: lowercase, split on whitespace and punctuation
            text = text.ToLowerInvariant();

            // Split on whitespace and common punctuation
            var regex = new Regex(@"[\w]+|[^\w\s]");
            var matches = regex.Matches(text);

            var tokens = new List<string>();
            foreach (Match match in matches)
            {
                tokens.Add(match.Value);
            }

            return tokens;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

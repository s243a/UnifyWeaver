import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from unifyweaver.targets.python_runtime.onnx_embedding import OnnxEmbeddingProvider

def test_onnx():
    model_path = "models/model.onnx"
    vocab_path = "models/vocab.txt"
    
    if not os.path.exists(model_path):
        print(f"Skipping: {model_path} not found")
        return

    print("Loading ONNX model...")
    embedder = OnnxEmbeddingProvider(model_path, vocab_path)
    
    text1 = "This is a test sentence."
    text2 = "This is a similar test sentence."
    text3 = "Something completely different."
    
    print("Generating embeddings...")
    vec1 = embedder.get_embedding(text1)
    vec2 = embedder.get_embedding(text2)
    vec3 = embedder.get_embedding(text3)
    
    print(f"Embedding shape: {vec1.shape}")
    assert vec1.shape == (384,)
    
    # Cosine similarity
    sim12 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    sim13 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    
    print(f"Sim(1,2): {sim12:.4f}")
    print(f"Sim(1,3): {sim13:.4f}")
    
    assert sim12 > sim13, "Similar sentences should have higher similarity"
    print("Test passed!")

if __name__ == "__main__":
    test_onnx()

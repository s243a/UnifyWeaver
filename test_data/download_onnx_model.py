#!/usr/bin/env python3
"""
Download and convert all-MiniLM-L6-v2 model to ONNX format.
"""

import os
import sys

def download_model():
    """Download and convert model using optimum."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        output_dir = "../models"

        print(f"Downloading model: {model_name}")
        print(f"Output directory: {output_dir}")
        print()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Download and convert model to ONNX
        print("Step 1: Downloading and converting model to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True
        )

        # Download tokenizer
        print("Step 2: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save model
        print("Step 3: Saving model and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print()
        print("✓ Download complete!")
        print()
        print("Model files saved to:")
        print(f"  {output_dir}/model.onnx")
        print(f"  {output_dir}/tokenizer.json")
        print(f"  {output_dir}/tokenizer_config.json")
        print(f"  {output_dir}/vocab.txt")
        print()

        # Create symlink for compatibility
        onnx_src = os.path.join(output_dir, "model.onnx")
        onnx_dst = os.path.join(output_dir, "all-MiniLM-L6-v2.onnx")

        if os.path.exists(onnx_src) and not os.path.exists(onnx_dst):
            os.symlink("model.onnx", onnx_dst)
            print(f"Created symlink: {onnx_dst}")

        return True

    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print()
        print("Please install optimum:")
        print("  pip install optimum[onnxruntime]")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def download_simple():
    """Simple download without ONNX conversion."""
    try:
        from transformers import AutoTokenizer, AutoModel

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        output_dir = "../models"

        print(f"Downloading model (PyTorch): {model_name}")
        print(f"Output directory: {output_dir}")
        print()

        os.makedirs(output_dir, exist_ok=True)

        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)

        print("Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(output_dir)

        print()
        print("✓ PyTorch model downloaded!")
        print("Note: You'll need to convert to ONNX separately")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("all-MiniLM-L6-v2 Model Downloader")
    print("="*60)
    print()

    # Try ONNX conversion first
    if not download_model():
        print()
        print("Falling back to PyTorch download...")
        print()
        download_simple()

import numpy as np
import re
import os
from .embedding import IEmbeddingProvider

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class OnnxEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, model_path, vocab_path):
        if not ort:
            raise ImportError("onnxruntime not installed")
        self.sess = ort.InferenceSession(model_path)
        self.vocab = self._load_vocab(vocab_path)
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100

    def _load_vocab(self, path):
        vocab = {}
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        return vocab

    def get_embedding(self, text):
        tokens = self._tokenize(text)
        input_ids = [self.cls_token_id] + \
                    [self.vocab.get(t, self.unk_token_id) for t in tokens] + \
                    [self.sep_token_id]
        
        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        token_type_ids = np.zeros_like(input_ids)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
        output = self.sess.run(None, inputs)[0] # [1, seq_len, 384]
        return self._mean_pooling(output, attention_mask)[0]

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\w+|[^\w\s]', text)

    def _mean_pooling(self, token_embeddings, attention_mask):
        # token_embeddings: [batch, seq, dim]
        # attention_mask: [batch, seq]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

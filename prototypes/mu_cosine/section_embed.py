#!/usr/bin/env python3
"""e5 encoder for the EMBEDDING section-categorisation layer — the semantic escalation after exact + fuzzy
(pt_sections.py). Lexical fuzzy catches TYPOS (`Subtoipcs`); this catches SYNONYMS / PARAPHRASES a section
author might use that share no edit-distance with the canonical keyword (`Members`, `Narrower areas`,
`Parent concepts`, `Things filed here`). Kept in a SEPARATE module so `pt_sections.py` stays stdlib-only and
torch-free — `categorize(..., method="embedding", encoder=…)` takes the encoder as an injected callable.

The encoder mirrors `mu_attention.build_e5_tables`: the same frozen `intfloat/e5-small-v2`, the same
asymmetric `query:` / `passage:` prefixes (the section label is the query; the canonical relation exemplars
are passages), unit-normalised. Loads OFFLINE from the HF cache (no egress)."""
import functools
import os

E5_MODEL = "intfloat/e5-small-v2"


@functools.lru_cache(maxsize=2)
def _model(model_name=E5_MODEL, device=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")            # use the local cache only — no network
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def e5_encoder(model_name=E5_MODEL, device=None, batch_size=64):
    """Return a callable `encode(texts) -> [N, 384] unit-normed numpy array`. Texts are expected to already
    carry their `query: ` / `passage: ` prefix (the caller decides which is which). The model is loaded once
    and cached."""
    model = _model(model_name, device)

    def encode(texts):
        return model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)

    return encode


if __name__ == "__main__":                                 # smoke: encode a couple of phrases, print a cosine
    enc = e5_encoder()
    v = enc(["query: members of this category", "passage: members", "passage: broader categories"])
    print("cos(label, 'members')        =", float(v[1] @ v[0]))
    print("cos(label, 'broader cats')   =", float(v[2] @ v[0]))

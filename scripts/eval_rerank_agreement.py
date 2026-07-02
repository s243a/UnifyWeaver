#!/usr/bin/env python3
"""eval_rerank_agreement.py — reranking-as-eval + token-cost sample.

For N test bookmarks: μ-shortlist (federated engine in `mu` mode) → LLM rerank → aggregate:
  * displacement / top1_changed  — how much the LLM reorders μ. LOW = μ agrees with a strong judge = μ ranks
    well (a μ-quality signal that needs NO ground truth, works on any bookmark). Also predicts economics:
    low displacement ⇒ the agent has little to fix ⇒ fewer agent tokens.
  * total_tokens_est per rerank   — the reranker's own cost (feeds tokens-per-correct-filing).
  * (optional, with a true folder) — μ-top1 correct? LLM-top1 correct? — validates the LLM JUDGE against
    ground truth before we trust its displacement.

Usage:
  python3 eval_rerank_agreement.py --model models/pearltrees_federated_s243a.pkl \\
      --mu-ckpt prototypes/mu_cosine/model_prod.pt --provider claude --llm-model haiku --n 20
  # custom bookmarks (one per line; optional TAB true-folder-title for accuracy):
  python3 eval_rerank_agreement.py ... --queries my_bookmarks.tsv

NOTE: this spends LLM quota (N rerank calls). Start small.
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm_cli
from llm_reranker import rerank

# in-domain-ish default probes for the s243a taxonomy (no true folders → displacement/tokens only)
DEFAULT_QUERIES = [
    "quantum entanglement and photon polarization", "bash shell scripting for loops",
    "the theory of relativity explained", "machine learning gradient descent",
    "cryptography and secure key exchange", "climate change and carbon emissions",
    "noam chomsky on media and propaganda", "linux filesystem permissions",
    "probability and statistics fundamentals", "surveillance and privacy rights",
    "neural network backpropagation", "economic inequality and wage stagnation",
    "graph theory and network analysis", "philosophy of mind and consciousness",
    "renewable energy solar and wind", "cognitive dissonance in decision making",
    "the history of the roman empire", "python data structures and algorithms",
    "game theory nash equilibrium", "anthropology and human evolution",
]


def load_queries(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        out.append((parts[0], parts[1] if len(parts) > 1 else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="federated .pkl model")
    ap.add_argument("--mu-ckpt", required=True, help="mu_cosine checkpoint")
    ap.add_argument("--queries", default=None, help="TSV 'bookmark[<TAB>true_folder_title]' per line; else built-in set")
    ap.add_argument("--n", type=int, default=20, help="cap number of bookmarks")
    ap.add_argument("--top-k", type=int, default=15, help="μ shortlist size handed to the reranker")
    ap.add_argument("--provider", default="claude"); ap.add_argument("--llm-model", default="haiku")
    a = ap.parse_args()

    queries = load_queries(a.queries) if a.queries else [(q, None) for q in DEFAULT_QUERIES]
    queries = queries[:a.n]

    from infer_pearltrees_federated import FederatedInferenceEngine
    engine = FederatedInferenceEngine(Path(a.model), routing_method="mu", mu_ckpt=a.mu_ckpt)

    disp, top1c, toks = [], [], []
    mu_ok, llm_ok = [], []
    print(f"[RERANK-EVAL] {len(queries)} bookmarks | μ shortlist top-{a.top_k} → {a.provider}/{a.llm_model} rerank\n")
    for q, true_folder in queries:
        cands = engine.search(q, top_k=a.top_k)
        cd = [{"title": c.title, "tree_id": c.tree_id} for c in cands]
        if not cd:
            continue
        llm_cli.reset_usage()
        out = rerank(q, cd, provider=a.provider, model=a.llm_model)
        toks.append(llm_cli.get_usage()["total_tokens_est"])
        if out["parsed"]:
            disp.append(out["displacement"]); top1c.append(1 if out["top1_changed"] else 0)
        mu_top, llm_top = cd[0]["title"], (out["reranked"][0]["title"] if out["reranked"] else "")
        flag = ""
        if true_folder:
            mu_ok.append(1 if true_folder.lower() in mu_top.lower() else 0)
            llm_ok.append(1 if true_folder.lower() in llm_top.lower() else 0)
            flag = f"  [true~{true_folder}]"
        print(f"  disp={out['displacement']:.2f} moved_top1={out['top1_changed']!s:5}  μ#1={mu_top[:28]:28} → LLM#1={llm_top[:28]}{flag}")

    print("\n== AGGREGATE ==")
    if disp:
        print(f"  mean displacement   {st.mean(disp):.3f}   (0=LLM kept μ order; lower ⇒ μ ranks better)")
        print(f"  top1_changed rate   {st.mean(top1c):.1%}   (how often the LLM moved a different folder to #1)")
        print(f"  parsed              {len(disp)}/{len(queries)}")
    if toks:
        print(f"  tokens/rerank est   mean {int(st.mean(toks))}  total {sum(toks)}  (reranker's own cost)")
    if mu_ok:
        print(f"  μ-top1 correct      {st.mean(mu_ok):.1%}   |  LLM-top1 correct {st.mean(llm_ok):.1%}  "
              f"(ground-truth match; validates the judge)")
    print("\n  Read: low displacement + high parsed ⇒ μ already near the LLM's order ⇒ a separate reranker adds little "
          "and the agent has little to fix. High displacement ⇒ μ misorders ⇒ rerank/agent correction matters.")


if __name__ == "__main__":
    main()

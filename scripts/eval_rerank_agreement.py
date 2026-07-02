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
        out.append((parts[0], parts[1] if len(parts) > 1 else None, None))   # (bookmark, true_title, true_id=None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="federated .pkl model")
    ap.add_argument("--mu-ckpt", required=True, help="mu_cosine checkpoint")
    ap.add_argument("--queries", default=None, help="TSV 'bookmark[<TAB>true_folder_title]' per line; else built-in set")
    ap.add_argument("--sample-filing", action="store_true",
                    help="sample N REAL bookmarks (+ true folders) from the Pearltrees filing data — large N + accuracy")
    ap.add_argument("--trees", default=".local/data/pearltrees_api/trees",
                    help="Pearltrees trees dir for --sample-filing")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed for --sample-filing")
    ap.add_argument("--n", type=int, default=20, help="cap number of bookmarks")
    ap.add_argument("--top-k", type=int, default=15, help="μ shortlist size handed to the reranker")
    ap.add_argument("--provider", default="claude"); ap.add_argument("--llm-model", default="haiku")
    a = ap.parse_args()

    from infer_pearltrees_federated import FederatedInferenceEngine
    engine = FederatedInferenceEngine(Path(a.model), routing_method="mu", mu_ckpt=a.mu_ckpt)
    eng_ids = set(str(x) for x in engine.global_target_ids)      # folders the model actually knows

    if a.queries:
        queries = load_queries(a.queries)
    elif a.sample_filing:                                        # real bookmarks + true folders from filing data
        import random
        sys.path.insert(0, str(Path(__file__).parent.parent / "prototypes" / "mu_cosine"))
        from eval_filing import load_filing
        pairs, cand = load_filing(a.trees, min_bm=3)             # [(bm_text, folder_tid)], {folder_tid: title}
        rng = random.Random(a.seed); rng.shuffle(pairs)
        # condition on the true folder being IN the model — else it can't be a candidate (that's a coverage gap,
        # not a ranking miss). Measures μ's real recall@k, not the federated model's 53% folder coverage.
        queries = [(bm, cand.get(f), str(f)) for bm, f in pairs if cand.get(f) and str(f) in eng_ids]
        print(f"[SAMPLE] {len(pairs)} filing bookmarks; kept {len(queries)} whose true folder is in the model (seed {a.seed})")
    else:
        queries = [(q, None, None) for q in DEFAULT_QUERIES]
    queries = queries[:a.n]

    disp, top1c, toks = [], [], []
    mu_ok, llm_ok, recall = [], [], []
    print(f"[RERANK-EVAL] {len(queries)} bookmarks | μ shortlist top-{a.top_k} → {a.provider}/{a.llm_model} rerank\n")
    for q, true_folder, true_id in queries:
        cands = engine.search(q, top_k=a.top_k)
        cd = [{"title": c.title, "tree_id": str(c.tree_id)} for c in cands]
        if not cd:
            continue
        llm_cli.reset_usage()
        out = rerank(q, cd, provider=a.provider, model=a.llm_model)
        toks.append(llm_cli.get_usage()["total_tokens_est"])
        if out["parsed"]:
            disp.append(out["displacement"]); top1c.append(1 if out["top1_changed"] else 0)
        mu_top = cd[0]
        llm_top = out["reranked"][0] if out["reranked"] else {"title": "", "tree_id": ""}
        flag = ""
        if true_id:                                              # EXACT tree-id match (sample-filing)
            in_sl = any(c["tree_id"] == true_id for c in cd)     # is the true folder even in the shortlist?
            recall.append(1 if in_sl else 0)
            if in_sl:                                            # only score top1 where the folder IS a candidate
                mu_ok.append(1 if mu_top["tree_id"] == true_id else 0)
                llm_ok.append(1 if llm_top["tree_id"] == true_id else 0)
            flag = f"  [true {'IN' if in_sl else 'out'}]"
        elif true_folder:                                        # title-substring fallback (--queries)
            mu_ok.append(1 if true_folder.lower() in mu_top["title"].lower() else 0)
            llm_ok.append(1 if true_folder.lower() in llm_top["title"].lower() else 0)
        print(f"  disp={out['displacement']:.2f} moved_top1={out['top1_changed']!s:5}  μ#1={mu_top['title'][:26]:26} → LLM#1={llm_top['title'][:26]}{flag}")

    print("\n== AGGREGATE ==")
    if disp:
        print(f"  mean displacement   {st.mean(disp):.3f}   (0=LLM kept μ order; lower ⇒ μ ranks better)")
        print(f"  top1_changed rate   {st.mean(top1c):.1%}   (how often the LLM moved a different folder to #1)")
        print(f"  parsed              {len(disp)}/{len(queries)}")
    if toks:
        print(f"  tokens/rerank est   mean {int(st.mean(toks))}  total {sum(toks)}  (reranker's own cost)")
    if recall:
        print(f"  shortlist recall    {st.mean(recall):.1%}   (true folder present in the top-{a.top_k} μ shortlist — "
              f"top1 accuracy is bounded by this)")
    if mu_ok:
        print(f"  μ-top1 correct      {st.mean(mu_ok):.1%}   |  LLM-top1 correct {st.mean(llm_ok):.1%}  "
              f"(of the {len(mu_ok)} where the true folder IS in the shortlist — did rerank move it to #1?)")
    print("\n  Read: low displacement + high parsed ⇒ μ already near the LLM's order ⇒ a separate reranker adds little "
          "and the agent has little to fix. High displacement ⇒ μ misorders ⇒ rerank/agent correction matters.")


if __name__ == "__main__":
    main()

# Reproducibility — the multi-seed blend-judge experiments

Exact scripts used for `REPORT_blend_judge_sweep.md`. The `cd` is repo-relative (these live in `repro/`, one
level under `prototypes/mu_cosine/`).

## Prerequisites (working files in `/tmp/mu_data/`, produced by the pipeline)

These scripts consume intermediate artifacts; regenerate them first (from `prototypes/mu_cosine/`):

```bash
# 1. stratified Wikipedia pairs (both endpoints in the struct embedding)
python3 gen_wiki_relation_pairs.py --graph ../../data/benchmark/100k_cats/category_parent.tsv \
    --struct-emb /tmp/mu_data/struct_emb_recip.pt --per 220 --out /tmp/mu_data/wiki_rel_pairs.tsv
# 2. LLM (gpt-5.5-low) relation scores → graded round   (needs: source ~/.nvm/nvm.sh && nvm use 22)
python3 score_with_codex.py --pairs /tmp/mu_data/wiki_rel_pairs.tsv --out /tmp/mu_data/wiki_rel_scored.tsv
python3 convert_scored_to_graded.py --scored /tmp/mu_data/wiki_rel_scored.tsv --out /tmp/mu_data/wiki_rel_graded_pairs.tsv
# 3. blend-judge SYM rows + combined round (LLM + blend); see the report for the concat step
python3 emit_blend_judge.py --pairs /tmp/mu_data/wiki_rel_pairs.tsv --e5-cache /tmp/mu_data/wiki_rel_e5.pt \
    --model model_prod.pt --struct-emb /tmp/mu_data/struct_emb_recip.pt --lam 0.5 --out /tmp/mu_data/blend_judge_pairs.tsv
```

Also required: `model_prod.pt`, `struct_emb_recip.pt` (from `structural_embedding.py`), and — for
`eval_blend_prediction.py` — the held-out set + its e5 cache (`wiki_rel_heldout.tsv`, `wiki_rel_heldout_e5.pt`).

## Scripts
- **`blend_sweep.sh`** — the 3×3 (arm × seed) sweep on the *LLM-scored* SYM eval (the confounded metric).
- **`blend_multiseed.sh`** — trains A/B × seeds 2-3 with `--save`, then evals all checkpoints against the held-out
  **judge superposition** `T` (the *right* metric; the confirmed result).

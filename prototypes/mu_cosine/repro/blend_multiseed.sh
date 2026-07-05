#!/bin/bash
# Multi-seed confirmation of the held-out judge-superposition result: does B (blend) predict T better than
# A (LLM-only) and prod, consistently across seeds? Seed 1 checkpoints already exist (model_A_ft / model_blend_ft).
cd /home/s243a/Projects/UnifyWeaver/prototypes/mu_cosine
export PYTHONIOENCODING=utf-8
GRAPH=../../data/benchmark/100k_cats/category_parent.tsv
A_GRADED=/tmp/mu_data/wiki_rel_graded_pairs.tsv          # arm A: LLM only
B_GRADED=/tmp/mu_data/combined_graded_pairs.tsv          # arm B: LLM + blend(λ0.5)
for seed in 2 3; do
  UW_MU_GRAPH=$GRAPH python3 train_mu_attention.py --device cuda --init-from model_prod.pt \
    --graded "$A_GRADED" --pairs mu_pairs_scored_cumulative.tsv --pairs-corpus simplewiki \
    --steps 800 --quick-val --seed "$seed" --save /tmp/mu_data/model_A_s${seed}.pt > /tmp/mu_data/msA_s${seed}.log 2>&1
  echo "trained A seed $seed"
  UW_MU_GRAPH=$GRAPH python3 train_mu_attention.py --device cuda --init-from model_prod.pt \
    --graded "$B_GRADED" --pairs mu_pairs_scored_cumulative.tsv --pairs-corpus simplewiki \
    --steps 800 --quick-val --seed "$seed" --save /tmp/mu_data/model_B_s${seed}.pt > /tmp/mu_data/msB_s${seed}.log 2>&1
  echo "trained B seed $seed"
done
echo "=== EVAL: predict the held-out judge superposition T ==="
UW_MU_GRAPH=$GRAPH python3 eval_blend_prediction.py \
  --pairs /tmp/mu_data/wiki_rel_heldout.tsv --e5-cache /tmp/mu_data/wiki_rel_heldout_e5.pt \
  --struct-emb /tmp/mu_data/struct_emb_recip.pt --ref model_prod.pt --lam 0.5 \
  --models prod=model_prod.pt \
    A1=/tmp/mu_data/model_A_ft.pt B1=/tmp/mu_data/model_blend_ft.pt \
    A2=/tmp/mu_data/model_A_s2.pt B2=/tmp/mu_data/model_B_s2.pt \
    A3=/tmp/mu_data/model_A_s3.pt B3=/tmp/mu_data/model_B_s3.pt \
  2>&1 | grep -vE "UserWarning|warnings.warn"
echo "MULTISEED DONE"

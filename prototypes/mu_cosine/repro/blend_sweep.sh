#!/bin/bash
# Multi-seed test of the judge-superposition generality hypothesis: does the blend judge lift BASE simplewiki
# SYM held-out (never touched by the blend pairs), consistently across seeds, and does random-λ help?
cd /home/s243a/Projects/UnifyWeaver/prototypes/mu_cosine
export PYTHONIOENCODING=utf-8
GRAPH=../../data/benchmark/100k_cats/category_parent.tsv
RES=/tmp/mu_data/blend_sweep_results.tsv
printf "arm\tseed\tsym_corr\n" > "$RES"
declare -A ARMS=( [A_llm]=/tmp/mu_data/wiki_rel_graded_pairs.tsv \
                  [B_fixed]=/tmp/mu_data/combined_graded_pairs.tsv \
                  [C_random]=/tmp/mu_data/combined_random_pairs.tsv )
for seed in 1 2 3; do
  for arm in A_llm B_fixed C_random; do
    log=/tmp/mu_data/sweep_${arm}_s${seed}.log
    UW_MU_GRAPH=$GRAPH python3 train_mu_attention.py --device cuda --init-from model_prod.pt \
      --graded "${ARMS[$arm]}" --pairs mu_pairs_scored_cumulative.tsv --pairs-corpus simplewiki \
      --steps 600 --quick-val --seed "$seed" > "$log" 2>&1
    corr=$(awk -F'): ' '/\[SYM\]  held-out . corr/{split($2,a," "); print a[1]}' "$log")
    printf "%s\t%s\t%s\n" "$arm" "$seed" "$corr" >> "$RES"
    echo "done ${arm} s${seed}: corr=${corr}"
  done
done
echo "SWEEP DONE"

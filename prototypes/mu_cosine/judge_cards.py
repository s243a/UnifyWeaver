#!/usr/bin/env python3
"""Judge cards — the declared-identity descriptions behind the name-function judge conditioning
(REPORT_channel_campaign.md §5: schema + measured e5 hierarchy; §6: the migration they enable).

Each card is a structured identity rendered as natural language, embedded with frozen e5 — the same
e5-phrase-seed idiom as the operator anchors (anchored_basis.py / EMBED_EXEMPLARS). Measured properties the
schema exploits: TYPE is the coarsest partition (LLM vs structural vs human, 0.831), then vendor (0.956),
then architecture (+0.025 cross-vendor MoE affinity), then version/variant (0.957-0.968), then effort
(0.948, correctly subordinate). Spaces not dashes (dashes fragment on tokenizer boundaries).

Policies (§5): TYPE-first; effort in the description; TRUTHFULNESS — only vendor-confirmed metadata, omit
unknowns rather than guess (no parameter-count rumors); DECLARED vs MEASURED boundary — the card is the
prior (identity as declared); measured behavior (R, bias, format discipline) stays in calibration estimates
and is never fed back into the card.

  python3 judge_cards.py          # print the cards + pairwise e5 cosines (sanity check)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import JUDGES, build_e5_tables

# Cards for every JUDGES index, plus ready-to-onboard judges not yet in the table (they onboard at r=0 =
# pure name prior once given an index). Effort appears only where the scoring actually pinned it.
JUDGE_CARDS = {
    "haiku": "LLM judge anthropic claude haiku, small fast model tier",
    "graph": "deterministic structural judge, graph walk, no language model",
    "human": "human curator judge",
    "sonnet": "LLM judge anthropic claude sonnet, mid size model tier",
    "opus": "LLM judge anthropic claude opus, large model tier",
    "gemini": "LLM judge google gemini",
    "gpt-5.5-low": "LLM judge openai gpt 5.5, low reasoning effort",
    "blend": "composite dual judge, e5 embedding model and graph walk blend, symmetric association channel",
    "dir-blend": "composite blend judge, graph walk and LLM element and subcategory estimators, "
                 "direction superposition",
    # not in JUDGES yet — B2 step 3 onboarding candidate (card matches the §5 worked example):
    "gpt-5.6-luna": "LLM judge openai gpt 5.6 luna, mixture of experts architecture, low reasoning effort",
}

CARDS_CACHE = "/tmp/mu_data/judge_cards_e5.pt"


def judge_card_e5(names=None, cache_path=CARDS_CACHE):
    """[J, 384] frozen e5 card embeddings, one row per judge, in JUDGES index order by default.
    Passage-prefixed (cards are descriptions, not queries), unit-normed — build_e5_tables' convention."""
    if names is None:
        names = sorted(JUDGES, key=JUDGES.get)
    missing = [n for n in names if n not in JUDGE_CARDS]
    assert not missing, f"no card for judges: {missing} — add to JUDGE_CARDS (truthful metadata only)"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _, passage, idx = build_e5_tables(list(names), cache_path=cache_path,
                                      texts={n: JUDGE_CARDS[n] for n in names})
    return passage, list(names)


def main():
    names = sorted(JUDGES, key=JUDGES.get) + ["gpt-5.6-luna"]
    E, _ = judge_card_e5(names, cache_path="/tmp/mu_data/judge_cards_e5_all.pt")
    for n in names:
        print(f"{n:14s} {JUDGE_CARDS[n]}")
    C = (E @ E.t()).numpy()
    print("\npairwise e5 cosine:")
    print(" " * 14 + " ".join(f"{n[:9]:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:14s} " + " ".join(f"{C[i, j]:9.3f}" for j in range(len(names))))


if __name__ == "__main__":
    main()

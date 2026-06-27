#!/usr/bin/env python3
"""Anchored-basis attention over the relation space (DESIGN_inferred_operator_superposition.md §8/§8b) —
the open-set generalisation of the operator superposition.

The relation basis is two-part: **frozen, label-tied ANCHORS** (one per principle tag) ++ **K learnable
residual ATOMS** (the unnamed/out-of-set relations). Realised as ATTENTION (§1b): each basis entry is a
(key, value) pair —
  * query   = the μ-feature vector,
  * keyᵢ    = sets how much entry i fires:  wᵢ = softmax(q_proj(query)·keyᵀ)ᵢ,
  * valueᵢ  = entry i's contribution to the token:  token = Σ wᵢ·valueᵢ.

Anchors: VALUE frozen (seeded from the e5 phrase embedding — stable, interpretable), KEY learnable but
CALIBRATED by the anchor-confidence KL (labels teach *when* it fires; *what* it contributes stays pinned).
Atoms: key AND value learnable — they compete with the anchors in the same softmax for the residual mass.

Weights are a finite categorical on a simplex (no integration). This module is the basis; wiring it into the
trainer's blend path + the grow/prune controller are the next increments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchoredBasis(nn.Module):
    def __init__(self, anchor_values, n_atoms=5, d_query=6, d_k=64, atom_init=0.02):
        """anchor_values: [A, d_model] FROZEN value embeddings (e5 phrase seeds). n_atoms K: learnable atoms."""
        super().__init__()
        n_anchors, d_model = anchor_values.shape
        self.n_anchors, self.n_atoms, self.d_model = n_anchors, n_atoms, d_model
        self.register_buffer("anchor_values", anchor_values.clone())        # FROZEN (not a Parameter)
        self.anchor_keys = nn.Parameter(torch.randn(n_anchors, d_k) * atom_init)   # learnable, KL-calibrated
        self.atom_keys = nn.Parameter(torch.randn(max(n_atoms, 0), d_k) * atom_init)
        self.atom_values = nn.Parameter(torch.randn(max(n_atoms, 0), d_model) * atom_init)
        self.q_proj = nn.Linear(d_query, d_k)

    def _keys_values(self):
        if self.n_atoms:
            return (torch.cat([self.anchor_keys, self.atom_keys], 0),
                    torch.cat([self.anchor_values, self.atom_values], 0))
        return self.anchor_keys, self.anchor_values

    def forward(self, query):
        """query [B, d_query] → (token [B, d_model], weights [B, A+K] on the simplex)."""
        keys, values = self._keys_values()
        w = F.softmax(self.q_proj(query) @ keys.t(), dim=-1)            # finite categorical, sums to 1
        return w @ values, w

    def anchor_kl(self, w, anchor_target):
        """KL( renormalised anchor weights ‖ label-confidence target ) — pins the anchor block to the labels.
        anchor_target [B, A] is the confidence-calibrated distribution over anchors for labelled rows."""
        a = w[:, :self.n_anchors]
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        return F.kl_div((a + 1e-9).log(), anchor_target, reduction="batchmean")

    @torch.no_grad()
    def utilization(self, w):
        """Per-entry mean mass-share (the grow/prune signal, §8b). atom_mass near 0 ⇒ prunable;
        all-saturated + gap-open ⇒ grow."""
        m = w.mean(0)
        return {"anchor_mass": m[:self.n_anchors].tolist(),
                "atom_mass": m[self.n_anchors:].tolist()}

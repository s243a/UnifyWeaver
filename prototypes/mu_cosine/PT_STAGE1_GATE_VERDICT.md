# Pearltrees Stage-1 gate verdict (2026-08-09) — PAUSE concept training

Gate declared in pt_gap_stage1.py BEFORE measurement; decision from the printed diagnostics.

- G1 (field informative): PASS — mean gap 0.1177 (>0.05), nonzero 39.1% (>35%), p90 0.444.
  e5 0.3183 vs best-mu 0.3020 on the 1,702 visible queries, 132-folder catalog.
- G2 (concept-coherent failures): FAIL — top-gap queries concentrate on LEXICAL
  CONTAINMENT: the destination folder's name appears verbatim in the bookmark title
  ('Noam Chomsky (2011)...' -> 'Noam Chomsky'; '...WikiLeaks Grand Jury...' ->
  'WikiLeaks'; 'Wikipedia: Laura Poitras' -> 'Laura Poitras'). Surface phenomenon
  (naming conventions), not concept knowledge. diag2 (smear preview) not needed —
  diag1 is decisive for the pause branch.

Consequence (per the pre-declared rule): concept training on Pearltrees is PAUSED;
no harvest spend. The surface mechanism differs from SimpleMind's (terse leaves there,
name-containment here) but both are surface-class. Note: the deployed softmax4 gate
already covers exactly these cases via its e5 channel (it beats e5 CI-solid overall);
the mu channels' residual Pearltrees value is the NON-overlap cases.

Cross-corpus pattern now confirmed three ways: e5 wins where lexical overlap decides;
mu wins where structure decides; the gate arbitrates per query. Sealed 1,136 untouched.

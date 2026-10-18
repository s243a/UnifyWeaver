# P1 data-gate report — outcome: `blocked_no_eligible_v2_labels`

Date: 2026-08-01. Checked per the encoder-lane handoff ("if the gate is not met, report
what's missing; that report is itself the deliverable").

## What exists
- `PROCESS_EXPRESSION_P1_PREREG.json` + `PROTOCOL_process_expression_p1.md` — sealed,
  protocol sha `21adc652…` verified present.
- `p1_ledger_builder.py` — consumes an eligibility report; exits 2 `blocked_no_eligible_v2_labels`
  without one.
- Routed campaign data in `~/mu_data/`: `routed_tasks_t0.02_n10.jsonl`,
  `routed_picks_{haiku,sonnet}*.jsonl`, `routed_lin_chunk*.jsonl` — **all legacy**:
  manifest prefix `fcf5e1d6…` (the prereg's `legacy_sources.historical_manifest_prefix`),
  bare `{qid, pick}` rows, no schema field, no execution bundles.

## What the gate requires and is missing
1. **v2-labeled routed bundles** — `unifyweaver.routed-task.v2` / `routed-picks.v2` /
   `routed-execution-bundle.v1` under `pearltrees-public-only-v1` privacy policy. None exist
   in the repo or `~/mu_data`. The legacy picks are explicitly not admissible:
   `legacy_unbound_picks_authorized: false`, historical counts descriptive-only.
2. **The eligibility verifier** (producer of the
   `unifyweaver.process-expression-p1-eligibility.v1` report) — sol's deliverable per the
   standing division of labor; not present in the repo. Without it the ledger builder
   cannot run even if bundles existed.
3. **The eligibility report itself** — none found (repo + `~/mu_data`).

## What unblocking costs (owner decision, per protocol)
Protocol line 70: "Producing new labels is a separate, explicit spending decision." Concretely:
re-run routed judge campaigns under the v2 execution-bundle harness (`routed_execution.py`
schemas) so picks are bound to task/menu bytes, plus sol lands the eligibility verifier +
privacy classifier index. Neither is training-lane work.

## Action taken
Per the handoff's priority rule, falling back to the R10 judge-channel refinement bundle
(prompt-text judge cards, asymmetric channel dropout, slot-compat test), which runs on
existing campaign data and merged rulings.

# Pearltrees data — the truncation gap and where recovery lives

A short pointer. The **full harvester onboarding lives in the harvester repo**, not here.

## The RDF export is capped

Pearltrees RDF exports **silently truncate at ~24 MB / exactly 5004 `<pt:Tree>` elements** — a
server-side cap. The XML still closes cleanly, so there is no parse error and nothing looks wrong.
This made the API-only harvest look ~93% incomplete and multi-parent filing folders look like
20/880 — both artifacts of the truncation. Background write-up: `docs/pearltrees_data_completion.md`.

## Recovering the missing data

Where recovery tooling *may* be available (present only when the private harvester checkout exists;
paths relative to this repo root):

- **`.local/tools/browser-automation/`** — the private harvester repo (`s243a/pt-harvester`,
  nested under the gitignored `.local/`). Its onboarding doc,
  **`.local/tools/browser-automation/docs/ONBOARDING.md`**, is the full pipeline map:
  the credentialed API backfill (re-fetch the trees the export dropped), the data layout, and the
  contentType legend. Start there for anything harvest-related.
- **`scripts/build_pearltrees_dag.py`** (this repo) — the *assemble* step: unions the RDF exports +
  API trees + `api_tree_paths` into `assembled_dag.tsv` + `assembled_titles.tsv`, and emits the
  backfill queue. Its docstring explains the data flow. Read-only on inputs.

On a clone without `.local/`, the harvester and harvested data are absent; the assemble/consume code
in this repo (`scripts/`, `prototypes/mu_cosine/`) is present but has nothing to read.

## Interpreting harvested data (this repo)

- `prototypes/mu_cosine/SKILL_understand_pearltrees.md` — turn harvested trees into typed membership
  relations (the contentType typing + principal-parent / nested-set ordering).
- Consumers: `prototypes/mu_cosine/{sample_product_kalman_pearltrees_campaign,run_pearltrees_fusion,eval_pearltrees_filing,filing_ranker}.py`.

The current recovered state is **partial** (396/880 multi-parent folders) pending a chunked
re-export + API backfill.

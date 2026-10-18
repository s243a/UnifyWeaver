# Pearltrees data completion

The Pearltrees dataset is incomplete because the **RDF export silently truncates at ~24 MB / ~5004 trees**
(confirmed: `<pt:Tree>` count frozen at exactly 5004 across every `s243a` export Oct 2023 → Jan 2026 while
`<pt:PagePearl>` grew; the large export has 36.5% dangling `RefPearl` child refs; the smaller `s243a_groups`
export, under the cap, has 0 dangling). It is a silent server-side cap — the XML still closes cleanly, so no parse
error. This made the API-only harvest look 93% incomplete and Pearltrees filing folders look only 20/880
multi-parent; both were artifacts of the truncation.

## What we have now

`scripts/build_pearltrees_dag.py` unions **all** local sources — both RDF exports (`context/PT/*.rdf`), the API
trees (`.local/data/pearltrees_api/trees/`), `api_tree_paths_v*.jsonl`, and `reports/*.jsonl` — into:

- `.local/data/pearltrees_api/assembled_dag.tsv` — the multi-parent DAG (`child<TAB>parent`), **14,709 nodes /
  25,449 edges / 4,641 multi-parent**, and recovers **396/880 multi-parent *filing* folders** (vs 20 API-only).
- `.local/data/pearltrees_api/assembled_titles.tsv` — `id<TAB>title` at **99%** coverage (RDF Tree titles + RefPearl
  display titles for dangling children + API titles).
- `.local/data/pearltrees_api/harvest_queue_augmented.json` — the existing 699-tree queue UNION the **4,040 still-
  missing referenced trees** (4,739 total), in the schema the browser-automation harvester consumes.

Re-run `python3 scripts/build_pearltrees_dag.py` any time new data arrives to fold it in (read-only on inputs).

## Completing the data — two complementary paths

**1. API backfill (primary, automatable).** The 4,739-tree `harvest_queue_augmented.json` gives exact `tree_id`s +
URIs to fetch. Review it, swap it over `harvest_queue.json`, and run the browser-automation harvester
(`.local/tools/browser-automation/`, `SKILL_harvest_pearltrees.md`) on your logged-in `s243a` session (which can
read both accounts incl. the `/t/s243a/` group spaces). Rate-limited, so it's a background fill. Then re-run
`build_pearltrees_dag.py` to fold the fetched trees in.

**2. Chunked RDF re-export (supplementary breadth).** To fix the source truncation, re-export `s243a` in pieces
that each stay **under the ~24 MB / ~5004-tree cap** — e.g. export per top-level category / sub-space rather than
the whole account at once (the `s243a_groups` export proves under-cap exports come out complete). Drop the chunk
`.rdf` files in `context/PT/` and add them to the `RDF_FILES` list in `build_pearltrees_dag.py`, then re-run.

The two paths overlap and dedupe on numeric tree ID; do either or both. Progress is measured by re-running the
pipeline and watching **MISSING trees** fall toward 0.

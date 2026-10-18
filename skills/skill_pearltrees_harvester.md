# Skill: Pearltrees Harvester Orientation

Where the Pearltrees harvest/assemble/consume pieces live and which resource to use. The **full
harvester detail lives in the harvester repo** (`.local/tools/browser-automation/docs/ONBOARDING.md`);
this skill routes to it and to the UnifyWeaver-side pieces.

> This is an on-demand reference skill (load when the task calls for it), not an auto-run tool.

## Which resource for which task

| Task | Resource |
|---|---|
| Full harvester pipeline / (re)harvest | `.local/tools/browser-automation/docs/ONBOARDING.md` (PRIVATE, needs `.local`) |
| The truncation gap + recovery pointer | `docs/pearltrees_harvester_onboarding.md` |
| Interpret harvested data → typed relations | `prototypes/mu_cosine/SKILL_understand_pearltrees.md` + `parse_pearltrees.py` |
| Rebuild the DAG from all sources (assemble) | `scripts/build_pearltrees_dag.py` (its docstring) |
| Why the data looks incomplete | `docs/pearltrees_data_completion.md` |
| File a bookmark into the hierarchy | `skills/skill_bookmark_filing.md` |

## Where the data is (gitignored, under `.local/`)

- `.local/data/api_tree_paths_v8.jsonl` — canonical materialized paths (`path_ids` = root→node).
- `.local/data/pearltrees_api/trees/` — per-tree JSONs; pearls with `contentType==1` are the filed
  bookmarks (the filing ground truth for `eval_pearltrees_filing.py`).
- `.local/data/pearltrees_api/assembled_dag.tsv` (`parent<TAB>child`) + `assembled_titles.tsv` (`id<TAB>title`).
- `.local/data/pearltrees_api/pearltrees_api.db` (SQLite `trees`,`pearls`), `api_responses.db` (raw cache).

## Two non-obvious facts

1. **The RDF export truncates at ~24 MB / exactly 5004 trees** (silent server cap; XML still closes
   cleanly). The assembled DAG (union of RDF + API + path files) is the recovered view, and is still
   a *partial* recovery (396/880 multi-parent folders) pending a chunked re-export + API backfill.
   Never treat the API-only or RDF-only counts as ground truth.
2. **The harvester is private.** `.local/tools/browser-automation/` is its own git repo (remote
   `s243a/pt-harvester`), using session cookies, nested under the gitignored `.local/` so it never
   enters UnifyWeaver commits. On a clone without `.local`, harvesting and harvested data are absent.
   Never commit cookies, logs, or `.local/` contents to a public repo.

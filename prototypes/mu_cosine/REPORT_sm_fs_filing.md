# SimpleMind filesystem filing corpus — fresh exploratory filing data

**The owner's insight, validated:** the .smmx maps live inside a real Dropbox folder hierarchy,
so each map's path is a RECORDED FILING DECISION — map root ↔ parent directory, the same
single-principal-folder task as Pearltrees, on a corpus the eval program has never touched.

## Historical v1 corpus + discipline (sm_fs_filing.py)

4,385 maps scanned; the original coarse filter (any path component containing `private`) → 4,367
with a filing folder; catalog = 270 distinct directory names (313 dirs ≥3 maps, the PT min_bm
analog) → 1,272 eligible queries. A deterministic 508-query reserve (40%, seed 0) was selected
before scoring and digested in `~/mu_data/sm_fs_ledger.json` (`eb02e3694be3b71d`); this script
never scores it. Exploration split n=764.

The reserve is **unscored but transductive**, not confirmatory: the catalog was derived from all
placements and the query split is not destination-folder-node-disjoint. A future prospective
cohort is required for a clean confirmatory claim.

The original substring filter was also not privacy certification. `sm_fs_privacy.py` now builds
a local contextual index: ordinary `/ACCOUNT/SLUG/id…` root links are public evidence;
`/private/id…` is unknown rather than automatically private; and exact `private` topic containers
mark their specifically linked child maps. Corpus model review is operationally disabled until a
versioned, verified passing benchmark lock exists, so model output cannot enter an index or change
a classification in this version.
The local filing default excludes private while retaining unknown; `public-only` excludes both
private and unknown and fails closed while an explicit-private target reference is unresolved.
New privacy-bound runs use the separate `~/mu_data/sm_fs_ledger_v2.json` ledger by default.
Historical metrics below are not silently recomputed under that new population. Here `public`
means allowed by the owner's conservative metadata policy, not independently certified public
availability. Privacy propagation follows only references beneath an explicit `private` topic; it
does not recursively taint every ordinary link inside the referenced map. Unreadable maps and
unresolved references remain visible as unknown/review-perimeter counts.

The v2 freezer requires a content-integrity-bound, source-verified index and has no substring
fallback. Its `public-only` and `private-only` runs produce separate mode-0600, no-replace
ledgers, transductive training catalogs, and lineage targets; unknown rows are review-only. Each
catalog contains admitted destinations and their ancestors, not an outcome-independent evaluation
catalog. Exact map and ancestor paths remain the identities, duplicate titles do not collapse, and
private/unknown names are not carried in a public training bundle.

### Rule-only inventory on the current local snapshot

The final source-rederived v2 rule-only pass bound 4,385 map files
(`members_sha256=bec85e7439a56815f65272a4f4732ab4c242482d96b6c73076c95281d68be46f`)
and classified **3,612 public-policy / 7 private / 766 unknown**
(`index_sha256=475ccacc6087e7c21ee31309654b4636b87ba5249e71870039db42245ea8f537`).
No model or API was called, and an independent full rescan reproduced every row. The review
perimeter is explicit: 691 unreadable/structurally invalid maps (593 are zero-byte conflict-copy
placeholders and 98 fail topic-graph validation), 74 ambiguous `/private/id…` roots, and two
unresolved map references below explicit private markers.

### Local-model development evidence and candidate slate

Windows Ollama can run small local models on the GTX 1660 SUPER without consuming WSL's
constrained RAM. The synthetic benchmark contains 12 hand-designed policy cases, uses a strict
JSON schema and deterministic decoding, and never reads corpus data. Its repeated runs test
transport and label stability; they are not independent observations, and this post-rubric case
set does not estimate population accuracy or establish statistical significance.

Two model families remain worth further testing:

| role | local model | development result | disposition |
|---|---|---:|---|
| accuracy-first | Gemma 4 E2B IT QAT (`07ea59a47401`) | stable **10/12**, **6/7** model-review-eligible, explicit 2,048-token cap | retain |
| lighter second family | Qwen3 4B (`359d7dd4bcda`) | stable **9/12**, **5/7** model-review-eligible in the earlier complete-response run | retain; rerun under the normalized cap |

Neither passes the deliberately strict 12/12 development gate, so neither authorizes a corpus
pilot. Qwen2.5 7B's earlier 11-row outputs are not comparable because completion truncation was not
yet distinguished from schema failure. Qwen3 8B, Gemma 4 E4B QAT, Llama 3.1 8B, and DeepSeek-R1
8B were slower and/or less accurate on this smoke set, so they are not current hardware-constrained
front-runners. Phi-3's earlier three-card check was anecdotal and supplies no acceptance evidence.
Benchmark v2 now fixes the output budget, rejects length-truncated completions, captures timing and
token metadata, and binds the exact model/source identities before and after every run.

Both retained results comprise three deterministic stability repetitions over the same synthetic
case set (`case_set_sha256=1e883c3beda4500caf84edef4942ccf82d111f675372cefc7d9d7aeb0dbe6955`)
and prompt (`prompt_sha256=6a744904779617448e48c86b4783f91aa9beb8b6bd66f75f30ff065dc8edc6bf`).
The sealed local-only artifact digests are
`6a702c0769cb52def78e5d84ac3da6fde2cbab6feb4fc2c40dc36e8e9e74ecf7`
(Gemma) and
`8066607ee3e8ae6f818ec4f0f3ba9b1e8a33540df96f7ac48bb7d5e5540a15c0`
(Qwen). The Qwen artifact predates benchmark v2's explicit normalized output cap, so it remains
development evidence and must be rerun under v2 before a family comparison.

### Proposed two-family advisory ranker (not implemented)

Gemma and Qwen can be retained as distinct advisory sources for ranking the rule-`unknown` review
queue. Each would emit an ordinal access-control score under the same frozen rubric; those
self-reported scores are not probabilities. Freeze the owner-label estimand as binary private
handling versus public-policy handling; unresolved reviews remain unresolved and do not silently
become negatives. Select the untouched outer lineage sample independently of either model's score.
If score-guided sampling is ever used to save review effort, seal its inclusion probabilities and
use the corresponding weighted estimand so verification bias is visible.

Use lineage-blocked cross-fitting: fit each calibrator on inner-training lineages, produce
out-of-fold categorical predictions for the joint head, then refit the calibrators on all
outer-training lineages and evaluate the frozen stack exactly once on untouched outer lineages.
Freeze precision at a stated review budget as the primary queue-ranking metric; report recall at
that budget, average precision, per-source separability and correlation, log loss, stated-bin ECE,
and margin-gated AURC as secondary measures with a paired lineage-block bootstrap.

Compare Gemma-only, Qwen-only, a prior-corrected PoE, and the learned joint combiner on identical
Gemma/Qwen inputs; add rule evidence to the joint head only as a separate ablation. The
prior-corrected PoE combines calibrated posteriors as
`p_G(y) p_Q(y) / pi(y)` and then renormalizes (equivalently, combine two likelihood factors with
one class prior). Estimate `pi(y)` from only the corresponding inner-training lineages, then refit
it on all outer-training lineages; preregister smoothing and fail closed when any fold lacks class
support. A raw posterior-product can be reported only as a clearly named naive control;
neither control multiplies raw ordinal scores. PoE is not the primary method: the models share a
prompt and likely correlated training evidence, so treating them as independent can double-count
confidence. Regardless of ranking quality, model output remains advisory and cannot change privacy
state or authorize publication.

Internal synthetic benchmarking is loopback-Ollama-only: the bridge rejects remote,
credential-bearing, proxied, or disguised endpoints and does not pull absent models. The corpus
index CLI is rule-only and rejects all model providers because no verified passing benchmark lock
has been implemented. Hosted review would require a separate owner-approved redacted export and
is not implemented here. Loopback transport alone does not prove local execution: before any
future private-data pilot, the Ollama server must also run with cloud features disabled
(`disable_ollama_cloud: true` or `OLLAMA_NO_CLOUD=1`) and the pilot must verify that setting.

## Result (exploration split, e5 ranking)

| corpus | catalog | R@1 | MRR | R@50 |
|---|---|---|---|---|
| **SM-FS filing** | 270 | **0.478** | **0.560** | 0.819 |
| PT filing (standing) | 335 | 0.203 | 0.291 | — |
| SM in-map parent | 200 | 0.180 | 0.320 | — |

## Interpretation (revises the cross-corpus claim)

Diagnostic: exact title==dir only 1.3%, but **78% substring overlap** — maps are topic-named and
filed in topic-named folders, so the query→folder semantic gap is small. Pearltrees is hard
because bookmarks are wild page titles far from folder vocabulary. Refinement of
REPORT_wiki_multiparent's conclusion: task SHAPE (one principal parent) sets the metric
structure, but the QUERY→FOLDER SEMANTIC DISTANCE distribution sets the difficulty level —
PT ≈ SM-in-map (wild/deep queries), SM-FS much easier (topic-identity queries).

## Uses

1. **Training corpus:** thousands of real filing decisions with materialized directory paths — the
   LINEAGE target-factory shape (`lineage(fs, decay=…)` as a process expression), 5.5× the 799
   campaign rows under the historical count, available while the harvester drains and P1 awaits
   v2 bundles. Training must bind the selected privacy-index view.
2. **OPENQ-012 cross-corpus arm:** a fourth task-matched corpus (PT / SM-in-map / wiki / SM-FS).
3. **Unscored transductive reserve:** retain the 508 rows for one frozen evaluation, while making
   no node-disjoint or confirmatory claim.

Caveats: exploration-split numbers are descriptive (single seed, no CIs); catalog uses directory
NAMES (duplicates title-equivalent); depth-0 maps (no filing folder) excluded; per-item data
stays in `~/mu_data` (paths/names may be personal), repo carries aggregates only.

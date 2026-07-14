# Repeated-judge candidate capacity — structural no-spend preflight

## Bottom line

The repository-owned structural preflight **blocks the candidate builder at every registered campaign size**.
It consumed the two frozen graphs but no scores, residuals, historical labels, Nomic embeddings, or judge calls.

The preregistration defines `source_component` as an undirected graph connected component and caps every such
source at 10% of a corpus campaign.  Selected campaign components are endpoint-disjoint.  Charging only one
distinct endpoint to each candidate's declared source gives the deliberately loose necessary bound

```text
U_1(G) = sum_s min(n_s, max(1, floor(0.10 G))).
```

This bound is valid even under the preregistration's permissive reading in which the distant negative may be
disconnected from the anchor in the canonical graph.  If the shortest descendant-to-root hops for all three
rows are instead finite in that same graph, their paths through the shared descendant put all four distinct
endpoints in one weak component and give the sharper sensitivity

```text
U_4(G) = sum_s min(floor(n_s / 4), max(1, floor(0.10 G))).
```

The candidate schema currently permits both a finite distant hop and `anchor_distant_disconnected=true`; a
future builder must resolve that ambiguity and reject combinations impossible in the graph it actually uses.
The blocking conclusion here does not rely on that resolution.  Both bounds ignore the 32 balance cells, hop
constraints, degree matching, graph/Nomic agreement, historical exclusions, normalized-title aliases, and
actual packing conflicts.  With source-component identities frozen before endpoint exclusions, those later
filters can only lower capacity.  Thus `U_1(G) < G` is a conclusive fail-closed result, not a power estimate.

## Frozen graph inputs

The command uses the same canonical loaders as the covariance work.  Paths are operational and are absent
from the scientific JSON; content is bound by byte count and SHA-256.

| corpus | retained graph | bytes | SHA-256 |
|---|---|---:|---|
| exploratory | SimpleWiki `100k_cats/category_parent.tsv` | 10,126,922 | `4881beedfd876e3abb9f1783cbc3fb8a7350e108e3f531cc4de28ef9956dc8ec` |
| exploratory identity | `100k_cats/metadata.json` | 308 | `15a632799adebd8b4f736c8420add8969bb5e5e4f20aa21ecaae9e2d95a61577` |
| fresh | enwiki scoped LMDB `data.mdb`, retained under `Behavior` | 1,319,403,520 | `3bcfe59a3f85870f377fad1ea77547f7c3566370f6172e27748f4f7ceba5d690` |

The fresh loader also binds the exploratory graph because it removes SimpleWiki-title overlap.  LMDB
`lock.mdb` is runtime state and is deliberately excluded.

## Capacity result

The canonical undirected graph removes self-loops and collapses reciprocal/duplicate direction while retaining
all loader nodes.  Edge counts below are unique undirected non-self edges.

| corpus | nodes | edges | connected components | largest component | largest fraction |
|---|---:|---:|---:|---:|---:|
| exploratory | 84,136 | 196,876 | 34 | 83,194 | 98.88% |
| fresh | 75,901 | 99,971 | 1 | 75,901 | 100% |

| `G` per corpus | per-source cap | exploratory `U_1(G)` | fresh `U_1(G)` | joint gate |
|---:|---:|---:|---:|---|
| 160 | 16 | 143 | 16 | fail |
| 320 | 32 | 194 | 32 | fail |
| 512 | 51 | 251 | 51 | fail |
| 800 | 80 | 335 | 80 | fail |

The sharper same-source four-endpoint sensitivity is 58/93/131/189 for exploratory and 16/32/51/80 for
fresh at `G=160/320/512/800`.  It is not needed for the decision.

Even the smallest `G=160` is impossible.  The fresh result follows immediately from its one connected
component: the source cap permits at most 10% of the requested campaign.  The exploratory graph has a giant
component plus only small alternatives, so increasing `G` does not repair the deficit.

## Why the pipeline stops here

Generating a larger Nomic cache, enumerating triples, or building the historical inventory cannot rescue this
gate; each would only filter endpoints within the source components frozen by this audit.  Historical removal
must not recompute and split those source components, because that would silently redefine the concentration
unit.  The preflight therefore stops
before those operations.  In particular:

- no candidate pool or endpoint was selected;
- no model was loaded and no embedding was generated;
- no attempted or successful historical score file was consumed;
- no prompt, judge, or provider was contacted; and
- all campaign, covariance, QR, and CUDA authorization fields remain false.

The audit also found an exact-byte contract mismatch before any cache regeneration: the Nomic cache builder
uses the prefix `"clustering: "` including its trailing U+0020 space, while the selector had frozen the display
token without that space.  The selector now binds the actual cache-input bytes.  This correction changes no
embedding or outcome in this PR because Nomic was not run.

## Alternatives and disposition

| alternative | disposition | reason |
|---|---|---|
| silently call top-level branches “connected components” | reject | changes the preregistered dependency unit while hiding the change |
| relax or drop the 10% cap after seeing capacity | reject | converts a frozen dependence safeguard into a convenience choice |
| build the Nomic cache and candidate pool anyway | reject | spends compute after a mathematically conclusive upstream failure |
| switch corpora to graphs with many connected components | defer | changes the scientific population and needs its own amendment |
| define deterministic exclusive branch/source groups | candidate amendment | potentially feasible, but must name them as source groups, handle multi-branch nodes outcome-blindly, and retain true connected-component diagnostics |
| retain one giant source and model/resample its dependence explicitly | defer | requires a new power model, fold contract, and effective-sample-size justification |

**Subsequent result:** `REPORT_repeated_judge_source_regions.md` tested a stricter deterministic exclusive
source-region family with exact connected count and three-hop-safe cores.  No frozen `K={64,96,128}` passed:
hard local-support isolation removed too much core mass, and exploratory also failed the larger-`G`
four-endpoint capacity bounds.  The pipeline therefore remains blocked.  The preferred next fork is a
prospectively powered dependence-aware source design, not post-hoc relaxation of either audit.

Only after a replacement necessary gate passes should work resume in this order: construct the attempted-input historical
inventory, enumerate the structural universe, generate a revision-pinned Nomic cache over that universe,
freeze agreement/degree/tag rules, and run exact packability checks for every registered `G`.

## Reproduction

```bash
python prototypes/mu_cosine/run_repeated_judge_candidate_capacity.py \
  --artifact-repo /path/to/UnifyWeaver-with-ignored-graph-artifacts \
  --out /tmp/repeated_judge_candidate_capacity.json
```

For the frozen inputs, the default command writes the complete audit and exits with status 2.  This makes a
shell caller fail closed without needing to remember an extra flag.  Explicit reporting workflows may pass
`--audit-only` to return zero after writing the same blocked JSON; they must then inspect
`decision.candidate_builder_must_stop`.  Pipeline automation should not use `--audit-only`.  Exit 2 is also
used by `argparse` for command-line errors, so the presence of a completed artifact and its decision field
distinguish a capacity block from invalid invocation.  The tracked reproduction artifact is
`repro/repeated_judge_candidate_capacity/summary.json`: 14,322 bytes, SHA-256
`403b1a59b6e53c87f881dad65ab6af2f401e046c0c7009a1efd220eb3899cdd8`.  The focused capacity suite passes
24 tests; the broader repeated-judge regression set passes 92 tests.

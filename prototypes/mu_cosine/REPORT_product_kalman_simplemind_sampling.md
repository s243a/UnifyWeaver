# SimpleMind Product-Kalman Campaign Sampling

Status: pre-scoring exploratory artifact. No judge labels or Product-Kalman result were produced in this run.

## Primary View

The primary SimpleMind view is the plain parent hierarchy within each map. Blank containers are bypassed, but
`See Also`, `Via Link`, `Related`, super-category, navigation, wiki-anchor, explicit-relation, and cross-map edges
do not enter the primary lineage. Structural labels are classified case-insensitively because the source maps use
variants such as `super Categories` and `via link`.

The primary classifier also treats `subtopics` as a pass-through container, `super Topic` as a secondary
super-category container, and `Root Node` as an organizational sentinel. These labels were found during the
pre-scoring title audit and are removed before candidate pairs are formed.

Only chains ending at the map's content root are retained. A private topic removes its entire raw subtree before
endpoint metadata is assembled. Maps with private, sentinel, or conflicted-copy roots are excluded. Repeated
consecutive title identities collapse; nonconsecutive repeats and endpoint pairs with conflicting direction or hop
across path observations are excluded rather than resolved post hoc.

## Source

The pilot uses the six systems-theory maps available in `prototypes/mu_cosine/context/`.

| map | bytes | SHA-256 |
| --- | ---: | --- |
| `Chaos theory.smmx` | 3,341 | `95f64197b7923c0ce7da41c78732b3f073cacc49731f34cd76d794219c2135f6` |
| `Complex systems theory.smmx` | 4,536 | `7ff88d4f27add6853affaf81b620545dad0ecd7b2d00b00f07ed0d5b4e0c46aa` |
| `Cybernetics.smmx` | 2,983 | `612f7612cffb6487fbe124ba914d4ce0d38d02e40f37f15a14bd2be068be16a8` |
| `Dynamical systems.smmx` | 5,108 | `12eb0282b5b47f0e4d57fc59091f1684034a71c8fda412d028abbc3ea891588c` |
| `Engineering.smmx` | 23,846 | `49d9c6963c9d4d46e48d54892dc9dd6f82db8ce59c8539cc17b0bef44f1e32c1` |
| `LTI System Theory.smmx` | 65,167 | `9addc45279a0f50e12616123f6da5a4712d92ee719177cc26ac76905d1a7b5dc` |

Command:

```bash
python3 prototypes/mu_cosine/sample_product_kalman_simplemind_campaign.py \
  --maps \
    prototypes/mu_cosine/context/Chaos\ theory.smmx \
    prototypes/mu_cosine/context/Complex\ systems\ theory.smmx \
    prototypes/mu_cosine/context/Cybernetics.smmx \
    prototypes/mu_cosine/context/Dynamical\ systems.smmx \
    prototypes/mu_cosine/context/Engineering.smmx \
    prototypes/mu_cosine/context/LTI\ System\ Theory.smmx \
  --pairs 250 --hmax 5 --seed 0 \
  --pairs-tsv /tmp/mu_data/simplemind_campaign_pairs_unscored.tsv \
  --score-in /tmp/mu_data/simplemind_campaign_score_in_unscored.tsv \
  --manifest /tmp/mu_data/simplemind_campaign_manifest_unscored.json
```

## Source Audit

| measurement | value |
| --- | ---: |
| maps retained | 6 of 6 |
| raw topics | 685 |
| primary-view real topics | 513 |
| structural containers | 164 |
| organizational sentinel topics | 5 |
| content-rooted path records | 361 |
| cross-map link annotations counted in the source audit, not used as edges | 150 |
| explicit relations excluded from the primary view | 7 |
| wiki anchors excluded as nodes, retained as endpoint aliases | 3 |
| see-also ancestry rejections | 62 |
| super-category ancestry rejections | 34 |
| endpoint pairs rejected for direction conflict | 1 |
| endpoint pairs rejected for hop conflict | 11 |

The eligible pools after conflict removal contain 329, 292, 241, 169, and 104 pairs at hops 1 through 5.
No structural endpoint (`via link`, `see also`, `related`, or super-category label) remains in the selected table.

## Sample

The emitted table contains 250 unique unordered endpoint pairs, with 50 pairs at each hop from 1 through 5.
Sampling is deterministic and round-robins across maps while a map has eligible candidates.

| hop | eligible pool | maps represented | largest map contribution |
| ---: | ---: | ---: | ---: |
| 1 | 329 | 6 | 10 |
| 2 | 292 | 6 | 11 |
| 3 | 241 | 6 | 25 |
| 4 | 169 | 5 | 36 |
| 5 | 104 | 5 | 40 |

Engineering contributes 40 of the 50 hop-5 rows. This concentration is a property of the available clean deep
paths, not a confidence signal. Later evaluation must report per-map sensitivity and must not interpret a pooled
deep-hop effect as map-invariant.

The selected rows contain 174 unique endpoint identities, 174 unique normalized titles, and 174 unique raw titles.
Their identity overlay contains 148 Pearltrees slugs, two raw-title aliases, and one selected enwiki alias.
The deterministic formatting audit found no duplicate normalized-title groups or encoding defects. That audit is
not a spelling checker: raw typos such as `Differentail Equation` remain deliberately unchanged. Pearltrees slugs,
enwiki aliases, raw topic IDs, source-map IDs, and title aliases are preserved for later identity closure and
audited-title sensitivity analysis. No semantic corrections were applied.

## Ephemeral Artifacts

| artifact | SHA-256 |
| --- | --- |
| `/tmp/mu_data/simplemind_campaign_pairs_unscored.tsv` | `9e53e73493a98d5361282ab1a5f8553a7ef2d0b196ba8ce09c3eee9651ce74e3` |
| `/tmp/mu_data/simplemind_campaign_score_in_unscored.tsv` | `79fa5044e825c4daf127b2f400d3a6da89c423120c324ea26a5c5aea9b8d5e3e` |
| `/tmp/mu_data/simplemind_campaign_manifest_unscored.json` | `cf4b7fc3cc33995e53796b49f596aafa625e064c19d922c5169fe035bcb9a033` |

These files are local and ephemeral. The committed sampler, source fingerprints, seed, and command are the durable
regeneration anchors.

## Interpretation Guardrails

- The sample is judge-ready, not scored evidence.
- Within-map principal paths do not establish that SimpleMind is more tree-like or calibrates better than enwiki.
- Raw typos are a title-channel property, not evidence of lateral semantic drift.
- Structural and cross-map relations remain secondary sensitivity views; they are not silently mixed into hop.
- Product-Kalman still requires identity-disjoint calibration/evaluation and comparison with the registered
  `JointPosterior` baseline on held-out NLL, calibration, and selective risk.

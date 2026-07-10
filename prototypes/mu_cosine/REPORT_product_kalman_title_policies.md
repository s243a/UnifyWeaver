# Product-Kalman Cross-Corpus Title Policies

Status: frozen pre-scoring sensitivity artifact. These policies were written before cross-corpus judge labels or
Product-Kalman evaluation results existed.

## Purpose

Raw titles remain the primary campaign view. The audited-title view asks a narrower matched-pair question: how much
of any observed corpus difference is attributable to unambiguous orthographic errors? It reuses every pair ID,
endpoint ID, hop, and graph-provenance field and changes only reviewed endpoint titles.

The materializer rejects a policy when the source pair-table hash changes, an endpoint is absent, or the complete
set of observed raw-title variants differs from the reviewed set. It emits the raw title beside the audited title
and records canonical-title collisions for later identity-disjoint splitting.

## Review Rule

- Correct only unambiguous spelling or truncation errors.
- Require a reviewed raw-title set and written evidence for every correction.
- Do not correct ambiguous semantics, brands, resources, organizational intent, capitalization alone, or style.
- Do not use judge labels, model residuals, hop effects, or downstream gains to choose corrections.
- Keep structural cleanup upstream. `Root Node`, `subtopics`, and `super Topic` are graph scaffolding, not typo
  corrections; the SimpleMind primary-view adapter now removes or retypes them before sampling.

## Frozen Policies

| corpus | source pair SHA-256 | reviewed endpoints | corrected endpoints | affected pairs |
| --- | --- | ---: | ---: | ---: |
| enwiki | `e6bcb7c64704c9f69a854267648b9d3fb40bf7084cdbe35e51d1c5d5544a1e1b` | 413 | 0 | 0 |
| Pearltrees | `586c81c9065ef4f003f8039c78054f0f96bf35a0c311d1a8c3399533733f230f` | 284 | 36 | 53 |
| SimpleMind | `9e53e73493a98d5361282ab1a5f8553a7ef2d0b196ba8ce09c3eee9651ce74e3` | 174 | 14 | 36 |

Enwiki is an explicit unchanged control: the audited and raw prompts have identical titles.

### Pearltrees

The complete policy contains 36 spelling-only corrections and is authoritative. Representative examples are:

| endpoint ID | raw | audited |
| --- | --- | --- |
| `12759321` | 1984 immages | 1984 images |
| `11728045` | Anti-Democractic Think Tank Positions | Anti-Democratic Think Tank Positions |
| `10479932` | Copyright & Free Speach | Copyright & Free Speech |
| `10647433` | Electricity & Magnitism (Physics) | Electricity & Magnetism (Physics) |
| `10407535` | Net Nutrality and Society | Net Neutrality and Society |
| `10830626` | The Isreal Palistine Conflict | The Israel Palestine Conflict |
| `11392387` | Transitions to Totalitarism | Transitions to Totalitarianism |
| `10490516` | We the corperations | We the corporations |

Ambiguous possessives, grammar, capitalization, organizational labels, and unclear phrases remain raw. See
`title_policies/product_kalman_pearltrees_titles.json` for all reviewed endpoint IDs and evidence.

### SimpleMind

| endpoint ID | raw | audited |
| --- | --- | --- |
| `title:buffer amplier` | Buffer Amplier | Buffer Amplifier |
| `title:capacitor impedence` | Capacitor Impedence | Capacitor Impedance |
| `title:differentail equation` | Differentail Equation | Differential Equation |
| `title:floating steel ball valv` | floating steel ball valv | floating steel ball valve |
| `title:inductor impedence` | Inductor Impedence | Inductor Impedance |
| `title:ligtning arrestor` | Ligtning Arrestor | Lightning Arrestor |
| `title:nrtl (us designation for: national recoginized test laboratory)` | NRTL (... National Recoginized Test Laboratory) | NRTL (... National Recognized Test Laboratory) |
| `title:plc (programable logic controllers)` | PLC (Programable Logic Controllers) | PLC (Programmable Logic Controllers) |
| `title:process, chemical, mechanical & hydrualic engineering` | Process, Chemical, mechanical & Hydrualic Engineering | Process, Chemical, mechanical & Hydraulic Engineering |
| `title:rasberri pi` | Rasberri Pi | Raspberry Pi |
| `title:turnion ball valve` | Turnion Ball Valve | Trunnion Ball Valve |
| `title:ups (uninteruptible power supply)` | UPS (Uninteruptible Power Supply) | UPS (Uninterruptible Power Supply) |
| `title:values` | Values | Valves |
| `title:worm dive (worm weel)` | Worm Dive (worm weel) | Worm Drive (worm wheel) |

`Values -> Valves` has unusually strong internal evidence: its preserved Pearltrees slug is `valves`, and the error
was documented independently in `REPORT_two_judge_posterior.md`. Other corrections are restricted to obvious
orthography. Potentially noisy resources and organizational labels are intentionally not reinterpreted here.

## Identity Closure

Corrections do not merge graph IDs. They add canonical identities that the later split materializer must close.
In particular, SimpleMind endpoint IDs `title:ligtning arrestor` and `title:lightning arrestor` both map to
`lightning arrestor` and cannot cross calibration/evaluation boundaries. Pearltrees also contains three preexisting
same-title/different-ID groups (`art`, `society`, and `wikipedia c`) that require the same treatment.

## Commands

```bash
python3 prototypes/mu_cosine/materialize_product_kalman_title_sensitivity.py \
  --pairs /tmp/mu_data/simplemind_campaign_pairs_unscored.tsv \
  --policy prototypes/mu_cosine/title_policies/product_kalman_simplemind_titles.json \
  --audited-pairs /tmp/mu_data/simplemind_campaign_pairs_audited.tsv \
  --score-in /tmp/mu_data/simplemind_campaign_score_in_audited.tsv \
  --audit-manifest /tmp/mu_data/simplemind_campaign_title_audit.json
```

Use the corresponding enwiki or Pearltrees policy and paths for the other corpora.

## Ephemeral Outputs

| corpus | audited pairs SHA-256 | score input SHA-256 | audit manifest SHA-256 |
| --- | --- | --- | --- |
| enwiki | `5ed4ab5003c6aa7eb470b4eb9e60c2a03bbb46b1af4827998b060d22df55dfde` | `400fba7feab3028f8181177fafd36ad07269c3e121640bf50ca809c64ec04b7b` | `645b11a4e80784ba6ebb2fbfe32e61f3e88440a74b204b7c145ed8f42e028480` |
| Pearltrees | `f5ab3a837b04eb28768e383a7f6071953bdca7601e23834c0f7280a587b76d0f` | `dcbe79ca2b5fbde65af9294278d2cc373f602bb5c0a703e2bd02d104f8e112ff` | `1d4178af754e3b984cb3cc1511afb8261437754043e8a013092b36d10898a90c` |
| SimpleMind | `73021d58f393d78fac20adbe38af48806cb87c32f641af9d0e023cece02e5bdb` | `ba6a297beb62b48eeb5eb28c94148a19fee17f5b83ddccee7133ee270ce53315` | `8a0a1ed904b9663a2c8fced6f66e6dc486962c3ddf027a5b9a52ddfd0350da93` |

These files are local and ephemeral. The committed policies, source hashes, materializer, and tests are the durable
regeneration anchors.

## Interpretation Guardrails

- Compare raw and audited scores on exactly matched pair IDs.
- The delta estimates the effect of this narrow correction policy, not all possible title noise.
- A small delta does not prove the graph is clean; resources, cross-topic lineage, and ontology differences remain.
- A large delta is evidence of title-channel sensitivity, not proof that every correction recovers author intent.
- Product-Kalman remains a candidate/control until identity-disjoint held-out NLL, calibration, and margin-gated
  AURC beat the registered `JointPosterior` baseline.

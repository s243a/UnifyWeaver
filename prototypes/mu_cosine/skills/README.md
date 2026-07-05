# Tracked skill mirrors

`.claude/skills/` is gitignored in this repo, so version-controlled copies of project skills live here.

To **activate** one in your session, copy it into your Claude Code skills dir:

```bash
cp -r prototypes/mu_cosine/skills/uncertainty-estimation ~/.claude/skills/    # or the repo's .claude/skills/
```

## Skills
- **`uncertainty-estimation/`** — playbook for combining multiple signals into a μ / confidence / relation
  estimate (calibrated joint posterior + margin gate, *not* hand-set independent confidences). Points to
  [`../DESIGN_uncertainty_estimation_playbook.md`](../DESIGN_uncertainty_estimation_playbook.md).

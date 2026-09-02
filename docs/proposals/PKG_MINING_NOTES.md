<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Mining notes: `Pkg` (Puppy Linux) as a requirements source for `uw-resolve`

Bounded recon task referenced from `PACKAGE_MANAGER_LOGIC_PROPOSAL.md` §2d/§2e.
Goal: read Pkg's actual bash and map every real mechanism it encodes onto the
`uw-resolve` P0 predicate set defined in `GROK_PKG_RESOLVER_PROMPT.md`:
`package/2`, `depends/4`, `conflicts/3`, `base/1`, `installed/1`,
`requested/1`, and the four queries `resolve/2`, `resolve_layered/2`,
`layer_closure/2`, `removal_orphans/2` (plus `explain_blocked/2`). No code was
changed; this file is the only deliverable.

**Honesty note:** every claim below either cites `file:line`/function name in
the cloned tree, or is explicitly marked `(inference)`. Nothing is from
memory of Puppy/Pkg outside what was read this session.

## 1. Provenance

- **Source cloned:** `https://github.com/puppylinux-woof-CE/Pkg` (option 1 in
  the task, reachable on first try; options 2/3 were not needed).
- **Clone:** `git clone --depth 50`, into
  `/tmp/.../scratchpad/pkgmine/woof-ce` (outside the UnifyWeaver tree, not
  vendored).
- **HEAD of clone:** commit `d64aff6657ea67a0d2ba5e13f70ee8cd8a33d094`,
  authored `2021-10-04`. Most recent commit touching the main script:
  `df5e828c8627da5782f9fa96489411c3a6454082`, `2020-12-17` — consistent with
  "dormant since ~2021" in the task brief.
- **Shallow-clone caveat:** `--depth 50` means `git log` for the main script
  only reaches back to `2018-12-31` in this clone; it is *not* the true
  project start. The script's own header and inline date-tagged comments
  supply the real origin: `usr/sbin/pkg:1-25` — `Copyright (c) 2013 Puppy
  Linux Community`, author "Scott Jarvis (sc0ttman)", `APPVER="1.9.23"`,
  license GPL. Inline comments carry DDMMYY change-tags going back further
  still, e.g. `#130511` (13 May 2011, `usr/sbin/pkg:177`), `#110913`,
  `#260713`, `#080917` — i.e. the dependency/blacklist/layer logic mined below
  was accreted over roughly a decade of real user bug reports, not designed
  up front.
- **Script size:** `usr/sbin/pkg` is 7,950 lines (`wc -l`), by far the
  largest file in the repo; supporting tools are `pkgdialog` (830),
  `gpkgdialog` (1,264, GTK frontend), `ppa2pup`/`ppa2pup_gawk` (409+364,
  PPA→Puppy repo conversion), `slack2pup` (271, Slackware→Puppy repo
  conversion), `splitpkg`/`strippkg`/`buildpet` (124/128/195, PET
  build/split tooling), `makepkg-fast` (386). Total repo: 26,101 lines across
  the files `find` walked.
- **Docs read:** `README.md` (370 lines, the full command reference and
  feature list quoted below), `CHANGELOG` (351 lines, used to date specific
  features), `pet.specs` (a real PET metadata sample), `root/.pkg/sources-all`
  and `etc/pkg/sources-all` (repo-index format samples).
- **Areas not read in depth** (out of the bounded-task budget, flagged rather
  than silently skipped): `usr/sbin/gpkgdialog` (GTK UI, 1,264 lines — no new
  resolution logic expected, it's a frontend over the same `pkg` functions);
  `usr/sbin/ppa2pup*` / `usr/sbin/slack2pup` (repo-conversion scripts — noted
  by name only in §4); the `usr/share/buildpet/*.bp` build-recipe files
  (build-from-source specs, orthogonal to dependency *resolution*).

## 2. Command-vocabulary table

Transcribed from `README.md:183-298` (the `pkg help-all` usage text), grouped
by lifecycle. This is the candidate `pkg`-style CLI schema uw-resolve's plan
executor / CLI would express through the `cli_args` schema machinery.

| Lifecycle group | Commands (long\|short alias) | uw-resolve query coverage |
|---|---|---|
| **Global options** | `--ask\|-a`, `--quiet\|-q`, `--force\|-f`, `--no-colour\|-nc` | plan-executor flags, not resolver — see §7 |
| **Search / info** | `search\|s`, `search-all\|sa`, `names\|n`, `names-all\|na`, `names-exact\|ne`, `names-exact-all\|nea`, `all-pkgs`, `status\|ps\|PS`, `contents\|c`, `entry\|pe`, `installed\|pi`, `manpage\|man` | catalog lookup by name/bound field — `package/2` fact lookup, not a resolution query |
| **Install / remove** | `add\|a`, `get\|g` (alias of add), `get-only\|go`, `download\|d`, `install\|i`, `install-all\|ia`, `update\|pu`, `remove\|rm`, `uninstall\|u`, `uninstall-all\|ua`, `delete\|l`, `delete-all\|la` | `resolve/2` + plan-apply; `remove`/`uninstall` → `removal_orphans/2` |
| **Deps** | `deps\|e`, `deps-download\|ed`, `deps-all\|ea`, `deps-check\|ldd`, `list-deps\|le\|LE`, `what-needs\|wn` | `deps\|e` ≈ `resolve_layered` restricted to one pkg's closure; `list-deps` ≈ printing `depends/4` closure; `what-needs` ≈ reverse-dep query (no current uw-resolve predicate — see §3.6); `deps-check\|ldd` is **runtime linkage verification**, out of scope (§3.7) |
| **Blacklist / config** | `blacklist`, `whitelist`, `repo-pkg-scope`, `repo-dep-scope`, `bleeding-edge`, `rdep-check`, `autoclean`, `clean`, `show-config`, `workdir` | `blacklist`/`whitelist` ≈ an `excluded/1` fact set, orthogonal to `base/1` (§3.3); the rest are resolver *policy knobs* uw-resolve doesn't yet model (§3.8) |
| **Listing / status** | `list-downloaded\|ld`, `list-installed\|li\|LI`, `list-builtins\|lb`, `build-list\|pbl` | `installed/1` and `base/1` fact dumps |
| **Build from source** | `build-list\|pbl`, `build\|pb`, `pkg-combine\|pc`, `sfs-combine\|sc`, `repack\|pr`, `merge` | `pkg-combine`/`sfs-combine` ARE `layer_closure/2` materialized to a file — the central finding, §5 |
| **Repo management** | `repo\|r`, `repo-info\|ri`, `repo-update\|ru`, `repo-list\|rl`, `repo-file-list\|rfl`, `repo-convert\|rc`, `add-repo`, `rm-repo`, `dir2repo`, `add-source`, `update-sources` | catalog ingestion / federation — `repo-update` explicitly delegates to PPM, the resolver/plumbing boundary (§6) |
| **Format conversion** | `dir2pet`, `dir2sfs`, `dir2tgz`, `deb2pet`, `pet2sfs`, `pet2tgz`, `pet2txz`, `sfs2pet`, `tgz2pet`, `txz2pet`, `extract\|unpack`, `split` | out of scope — file-format plumbing, not resolution |
| **Misc / UX** | `welcome`, `show-config`, `version\|v`, `help\|h`, `help-all\|H`, `usage`, `examples\|ex`, `func-list` | CLI ergonomics, §8 |

Two commands imply capabilities uw-resolve's four queries don't currently
name: **`what-needs\|wn`** (reverse-dependency query over *installed*
packages — `list_dependents()`, `usr/sbin/pkg:6796`) and **`deps-check\|ldd`**
(runtime `ldd`-based missing-`.so` check, `pkg_ldd()`, `usr/sbin/pkg:6667`).
See §3.6–3.7.

## 3. Dependency heuristics — findings mapped to the design

### 3.1 The base/installed/user three-way split IS `base/1` — already-covered, with a wrinkle

`is_builtin_pkg()` (`usr/sbin/pkg:971-1002`), `is_usr_pkg()`
(`usr/sbin/pkg:1039-1070`), and the record files they read —
`${REPO_DIR}/woof-installed-packages` and `${REPO_DIR}/user-installed-packages`
— are exactly the `base/1` vs. non-base partition the proposal's §2d
describes, implemented in bash a decade before the Prolog framing. A third
file, `devx-only-installed-packages`, tracks the separately-loadable DEVX
(dev-tools) SFS the same way.

**Mapping:** `base(Pkg-Ver)` ← `woof-installed-packages` row;
`installed(Pkg-Ver)` ← union of all three files, computed at
`list_all_installed_pkgs()` (`usr/sbin/pkg:6167-6190`).

**Wrinkle our design should adopt (P0.5):** the "base" set is not static — it
is a function of *which SFS layers are currently mounted*. At startup
(`usr/sbin/pkg:176-184`) Pkg builds `layers-installed-packages` as
`woof-installed-packages` **plus** `devx-only-installed-packages` **only if**
DEVX looks loaded, and it tests that with `which gcc` (`usr/sbin/pkg:178`) —
i.e. "is gcc on PATH" is used as a proxy for "is the dev-tools SFS layer
mounted right now." This is real evidence for §2e's `layer_shadow` reason:
Pkg already treats "which packages are frozen" as **session-dependent on
mounted layers**, not a fixed manifest baked at build time. **Adopt:**
`base(Pkg-Ver, layer_shadow)` should be derived from *which layers are
currently mounted*, sourced from a layer-manifest predicate
(`layer_mounted/1`), not hardcoded once. Pkg's own heuristic for detecting
mount state (`which gcc`) is fragile and worth explicitly NOT copying — ours
should read the actual SFS mount table (or Woof-CE's own record of it), which
is a cleaner source than Pkg had available in pure bash.

### 3.2 Dependency closure walk — adopt the bound-depth iterative BFS pattern, note the wrinkle

`list_deps()` (`usr/sbin/pkg:6199-6361`) computes the transitive dependency
closure of a package with an explicit **iterative BFS bounded at 16 levels**
(`for i in 0 1 2 ... 15`, `usr/sbin/pkg:6268`), each level: strip already-done
packages (`DEP_DONE`, a visited-set file) and already-installed packages, then
expand the next level's deps. This is structurally identical to what
`layer_closure/2` needs to compute, but with two real-world compromises our
declarative closure removes:

- **Adopt (recognize, don't copy):** the depth bound exists because bash has
  no efficient recursion/fixpoint primitive; Prolog's `resolve_layered`/
  `layer_closure` should compute a true fixpoint (no artificial depth cap).
  Flag this explicitly as something Pkg got *wrong* under constraint — not a
  requirement to reproduce. Note for the P0 contract corpus: add a
  deep-dependency-chain test case (>16 levels) specifically because Pkg's own
  bound would silently truncate it — a scenario apt-based tooling wouldn't
  even think to test, but Puppy users with long chains (e.g. a desktop
  environment → toolkit → font stack → locale data) plausibly hit.
- **Adapt — already-covered with a wrinkle:** the visited-set (`DEP_DONE`)
  pattern is exactly the closure/memoization `layer_closure/2` needs
  internally; no new predicate required, just confirms the standard approach
  is right.

### 3.3 Blacklist/exclude is a THIRD partition, orthogonal to `base/1` — adopt

`is_blacklisted_pkg()` (`usr/sbin/pkg:919-930`) and the exclusion applied
inside every closure step (`grep -vE "'$PKG_BLACKLIST_REGEX'"`,
`usr/sbin/pkg:6253,6277,6342`) implement a **third category** the current
`GROK_PKG_RESOLVER_PROMPT.md` fact set doesn't name: packages the user (or
the distro, via `PKG_NAME_IGNORE` sourced from Woof-CE's own
`/root/.packages/PKGS_MANAGEMENT`, per `CHANGELOG:90`) has declared "never
auto-pull as a dependency," independent of whether it's base or installed.
Blacklisting is checked at *every* dependency-expansion step (never appears
in a closure) and also blocks fresh installs (`pkg_install()`,
`usr/sbin/pkg:4818`) — but, found while reading `pkg_uninstall()`
(`usr/sbin/pkg:5825-5828`), it **also blocks removal** of an
already-installed package that's since been blacklisted. That looks like an
unintended interaction (a user who blacklists a package they already have
installed can no longer `pkg uninstall` it without `--force`) — worth citing
as a **wrinkle to avoid**, not adopt: our design should make "excluded from
new installs" and "protected from removal" independently controllable facts.

**Adopt for P0.5:** an `excluded/1` fact, distinct from `base/1`, consulted
in `resolve`/`resolve_layered` candidate generation (never select an excluded
package to satisfy a dependency) but explicitly NOT consulted by
`removal_orphans` (removal safety should depend only on live reverse-deps,
never on exclusion state) — the opposite of what Pkg accidentally does.

### 3.4 Removal safety net — already-covered, but Pkg's provenance tracking for "installed as a dependency" is effectively dead code

`pkg_uninstall()` computes `list_dependents()` (`usr/sbin/pkg:6796-6852`) —
a real reverse-dependency scan across all repo files (`Packages-*`) restricted
to packages that are *actually installed* (`comm -12` against
`user-installed-packages`, `usr/sbin/pkg:6845-6846`) — and **refuses to
uninstall** if anything still depends on the target, unless `--force`
(`usr/sbin/pkg:5896-5909`). This is exactly `removal_orphans/2`'s safety
invariant ("no other installed package or request needs them") and confirms
the design is right: already-covered by §2d point 4.

**Honesty-flagged finding — a real gap in Pkg, worth noting as a
contradiction with an assumption a naive reading of Pkg might suggest:**
`pkg_remove()` (`usr/sbin/pkg:6009-6058`) *tries* to clean up "leftover
dependencies" by reading a per-package file matching `*_dep_list`
(`usr/sbin/pkg:6038`, `find $TMPDIR -iname "${1}*_dep_list"`) — but **no code
anywhere in the 7,950-line script ever writes a file matching that glob**
(confirmed via `grep -rn "_dep_list"` across the whole repo — the only hit is
the read site). The only similarly-named file, `${PKGNAME}_bgdep_list`
(`usr/sbin/pkg:5626,6546-6547`), is a session-scoped background-prefetch
cache written by `pkg_get()`, not a persistent per-package dependency
manifest. **Conclusion: Pkg has no persistent record of "which currently
installed packages were pulled in as a dependency of which request."**
`user-installed-packages` is a flat, undifferentiated list — direct user
requests and auto-pulled dependencies are indistinguishable once written.
`pkg remove`'s "leftover deps" cleanup pass is therefore vestigial/inert in
this version of the script; the *actual* safety net users get is the
reverse-dependency check in `pkg_uninstall` (§ above), which prevents bad
removals but never proactively offers orphan cleanup after a `remove`.

This is the single clearest point where **uw-resolve's design is already
ahead of Pkg**, not behind it: `requested/1` vs. `installed/1` as separate
facts (per `GROK_PKG_RESOLVER_PROMPT.md`) is precisely the distinction Pkg's
data model lacks, and `removal_orphans/2` computing "no request and no
surviving reverse-dep needs it" over that distinction is a real improvement,
not a re-derivation of something Pkg already had. **Adopt:** keep
`requested/1` as a first-class fact distinct from `installed/1` — cite this
finding as the reason it matters, not merely as good hygiene.

### 3.5 Repack/combine excludes exactly base ∪ (devx unless forced) ∪ blacklist — this IS `layer_closure` minus base, confirmed in code

`pkg_combine()`/`sfs-combine` (`usr/sbin/pkg:3730-3920`, dispatch table at
`usr/sbin/pkg:7821,7824`) builds a single artifact containing a package plus
its **non-base** dependency closure: builtins are skipped unless
`HIDE_BUILTINS=false` (`usr/sbin/pkg:3838`), DEVX packages are skipped unless
`--force` (`usr/sbin/pkg:3841`), blacklisted packages are always skipped
(`usr/sbin/pkg:3835`). This is §2d point 3 ("closure-minus-base IS the layer
manifest") independently arrived at in bash — **already-covered**, and the
exact filter conditions (base, devx-unless-forced, blacklist) are a
ready-made test-oracle for the P0 contract corpus's `layer_closure` cases.
See §5 for the SFS-specific packaging requirements this implies.

### 3.6 `what-needs` (reverse-dependency-of query) — adopt as a named predicate

`list_dependents()` (`usr/sbin/pkg:6796-6852`, exposed as `wn`) answers
"what installed packages need X" as a standalone, user-facing query — not
just an internal removal-safety check. `resolve_layered`/`removal_orphans`
compute reverse-deps internally but `GROK_PKG_RESOLVER_PROMPT.md` names no
predicate for exposing it directly. **Adopt:** add `dependents(Pkg,
Dependents)` (or similar) to the P0 predicate set — cheap (it's a subset of
what `removal_orphans` already computes) and it's explicitly one of the
"four questions Puppy users actually ask" per README's own quick-start flow
(`README.md:254`, listed as a top-level command, not buried).

### 3.7 `deps-check`/`ldd` — out of scope, name why precisely

`pkg_ldd()` (`usr/sbin/pkg:6667-6744`) runs the real `ldd(1)` against every
binary a package's file-manifest lists (`usr/sbin/pkg:6709-6722`) to find
missing shared libraries **that the declared dependency metadata didn't
catch** — a runtime/filesystem check, not a fact-base query. **Out of
scope** for a declarative resolver spec: it requires executing `ldd` against
real installed binaries on disk, which is executor-side verification, not
resolution. Worth a one-line mention in the phased plan as a *possible*
post-install executor step (verify the plan actually satisfied real linkage)
but explicitly not a `uw-resolve` predicate.

### 3.8 Policy knobs Pkg exposes that uw-resolve's plan/query layer doesn't yet name — adapt

`repo-pkg-scope one|all`, `repo-dep-scope one|all`, `bleeding-edge no|yes`,
`rdep-check no|yes` (README.md:271-274, backed by `set_pkg_scope()`,
`set_dep_scope()`, `set_bleeding_edge()`, `set_recursive_deps()` —
`usr/sbin/pkg:1216-1324`) are real user-facing resolution-policy switches:
whether to search only the current repo or federate across all repos when
resolving names/deps, and whether to prefer the newest version from *any*
repo vs. staying within one repo's version line. These map onto candidate
generation order in `resolve`/`resolve_layered` (§3 of
`GROK_PKG_RESOLVER_PROMPT.md` already specifies "prefer base-satisfied, then
highest version" as the determinism policy) — **adapt:** the policy needs a
knob for "highest version within the current/preferred repo" vs. "highest
version across all federated repos," because Puppy users apparently wanted
both and toggle it (`bleeding-edge`). Not a new predicate, but the candidate
ordering should be explicitly parameterized rather than a single fixed rule.

## 4. Record / repo-format notes — candidate fact-ingestion specs

### 4.1 The 12-field pipe-delimited package record (repo indexes AND installed-package DBs)

Confirmed by two independent sites: the `DB_ENTRY` constructed at
`usr/sbin/pkg:5195` and the shipped sample `pet.specs:1`:

```
Name|NameOnly|Version|Build|Category|SizeK|Dir|Filename|+Dep1,+Dep2,...|Description|DistroCompat|DistroCompatVersion
```

This single format is reused, unchanged, for: `Packages-*` repo-index files
(one per repo, e.g. `Packages-puppy-noarch-official`), `woof-installed-packages`
(base), `devx-installed-packages`/`devx-only-installed-packages`, and
`user-installed-packages`. All the `is_*_pkg()` lookup functions
(`usr/sbin/pkg:933-1136`) `cut -f1,2,8 -d'|'` or similar against these same
files. **Candidate ingestion spec:** a single fact-loader
`load_puppy_pkgdb(File) → package/2, depends/4` (deps field needs the
`+dep1,+dep2` → `depends(Name,_,Dep,any)` unpack seen in `list_deps()`,
`usr/sbin/pkg:6234-6239`, including the `:any`/`&ge`/`&lt` constraint-suffix
stripping at `usr/sbin/pkg:6160-6162` — note Pkg **discards** version
constraints on deps entirely rather than parsing them into `gte`/`range`;
our ingestion should do better and actually parse `&ge1.2.0` etc. into
`depends/4`'s `Constraint` argument instead of dropping it).

### 4.2 Repo-source federation list with per-repo fallback chains

`root/.pkg/sources-all` / `etc/pkg/sources-all` format:

```
name|ext|repo_file|mirror1|mirror2|mirror3|mirror4|fallback_repo_names...
```

e.g. `slacko|pet|Packages-puppy-slacko-official|<mirror urls>|||slacko14 noarch`
(`root/.pkg/sources-all:8`). The trailing field is a **space-separated,
priority-ordered fallback list of other repo names** to search if a package
isn't found in this one. **Candidate for the catalog design:** the GP-LMDB
catalog store (proposal §2c) should model repos as a small DAG/priority chain
(`repo_fallback(Repo, FallbackRepo, Priority)`), not a flat "all repos"
bag — this is a real federation topology Puppy repos actually use (e.g.
`slackware14.1` → `slackware14.1-patches` → `slackware14.1-salix` →
`slacko14.1` → `slacko14` → `noarch` → `common32`, `root/.pkg/sources-all:20`)
and it directly affects candidate ordering under `bleeding-edge`/`repo-scope`
(§3.8).

### 4.3 Name-alias table — adopt as a small `alias/2` fact set

`usr/sbin/pkg:227-236` hardcodes a name-normalization table, e.g.
`rxvt-unicode,urxvt,urxvt-unicode`, `gtk+,gtk+2*`, `dbus*,libdbus*,libdbus-glib*`,
`mesa,mesa_*,libgl1-mesa*,mesa-common*`. Real packages get renamed/split
across distro versions and repos disagree on canonical names; without this
table, dependency-name matching silently fails. **Adopt for P2 (catalog
ingestion):** an `alias(CanonicalName, MatchPattern)` fact set consulted
during candidate-name resolution before falling back to "not found" —
directly informed by this table, not invented from scratch.

### 4.4 Plugin hook points — adopt (lightweight) for the executor, not the resolver

`run_plugins()` (`usr/sbin/pkg:262-...`) defines hook names
`init|pre_install|post_install|pre_uninstall|post_uninstall|pre_build|post_build|exit`
(`usr/sbin/pkg:266`). This belongs on the **plan-executor** side (proposal
§4 constraint 4: "executors are dumb, apply-only"), not the resolver — but a
plan schema that names these same boundary points (so an executor can run
distro-specific hacks around each install/uninstall step, the way
`postinstall_hacks()` at `usr/sbin/pkg:5251` does for Puppy-specific fixups)
would make executors pluggable without touching the resolver spec.

## 5. SFS-workflow requirements for `layer_closure` output

Read from `pkg_combine()` (`usr/sbin/pkg:3730-3920+`) and the SFS build call
site (`usr/sbin/pkg:7109`, `mksquashfs "$SFS_DIR" "$SFS_NAME" -noappend`):

1. **Materialization is "download/unpack every closure member into one
   directory tree, then `mksquashfs` the tree."** `layer_closure/2`'s output
   (an ordered manifest) is consumed by, for each member: locate or download
   its package file, extract it into a shared build directory
   (`usr/sbin/pkg:3813-3871`), then squash the whole tree in one pass. The
   manifest itself doesn't need to carry file lists — just package identities
   — because extraction does that work.
2. **No file-conflict detection between closure members observed.** Each
   package's files get copied into the same build directory; a later
   package's file silently overwrites an earlier one with the same path (no
   check found at `usr/sbin/pkg:3861-3870` or nearby). **Out of scope for
   the resolver** (this is a packaging-time concern, not a selection
   concern) but worth flagging explicitly as a requirement the *executor*
   should add and Pkg doesn't have: `layer_closure`'s manifest should be
   sufficient for an executor to detect path collisions before squashing,
   even though Pkg itself never did.
3. **Ordering: the request itself is materialized before its dependencies**
   (`usr/sbin/pkg:3817-3823` copies the main package first, deps in the loop
   after, `usr/sbin/pkg:3830-3871`) — but since materialization here is "copy
   files into a merged tree, then squash once," order doesn't matter for
   *this* consumer. **This does not confirm or refute** whether
   `layer_closure`'s "dependencies before dependents" ordering requirement
   (`GROK_PKG_RESOLVER_PROMPT.md` line ~42) matters for some *other* SFS
   consumer (e.g. sequential `pkg_install` calls, where a package's
   post-install script might expect a dependency already present) —
   `pkg_install()` itself is called once per package with no
   observed ordering guarantee across a dependency set (inference: the
   `get_deps()` loop at `usr/sbin/pkg:6582-6652` installs in whatever order
   `DEPS_MISSING` iterates, not a topological order). **Adopt:** keep
   `layer_closure`'s dependency-before-dependent ordering guarantee anyway —
   it's strictly safer for executors that install sequentially, even though
   Pkg's own code doesn't enforce or need it for the squashfs path.
4. **The two SFS "flavors" (`pkg-combine` → `.pet`, `sfs-combine` → `.sfs`)
   share one code path**, gated only by `$COMBINE2SFS`
   (`usr/sbin/pkg:3788,7821,7824`) — i.e. `layer_closure`'s output format
   should be packaging-format-agnostic; the choice of `.pet` vs `.sfs` is a
   pure executor/output-format decision, not something the resolver needs to
   know about.

## 6. PPM interop boundary — already-covered (matches proposal §5 scope discipline)

`update_repo()` (`usr/sbin/pkg:2058-2100`) is the concrete boundary line: `pkg
repo-update` does **not** reimplement repo-index fetching. It shells out
directly to `/usr/local/petget/0setup` — PPM's own repo-sync tool — and
parses PPM's output text to report which repos updated
(`usr/sbin/pkg:2078-2094`). The comment at `usr/sbin/pkg:2069-2071` is
explicit about *why*: "petget wont accept $1 and only do that repo... but
petget code is way faster than mine... we're limited to updating only the
repos that petget supports." **This is exactly the resolver-vs-plumbing line
the proposal already draws** (§5 "a planner over existing package managers,"
constraint 5 "not a new package format, store, or Nix competitor"):
network repo-index syncing is explicitly PPM's job in Pkg's own design, never
reimplemented even though the author clearly wanted to (`README.md:26`:
"Puppy Package Manager (for `pkg repo-update` only)" is listed as Pkg's
*only* hard runtime dependency beyond bash/coreutils). **No new finding
needed here** — cite this as confirmation the proposal's scope line matches
what a decade of real Puppy tooling converged on independently.

## 7. CLI ergonomics worth stealing

- **`--ask` as a universal prefix flag**, not a per-command option: any
  destructive/mutating command becomes "confirm each item" by prepending
  `--ask` (`README.md:189`, e.g. `pkg -a e PKGNAME`, `pkg --ask add
  "firefox mplayer"`). Implemented as one global `$ASK` variable checked at
  every confirmation point (`read -n 1 CONFIRM`, e.g.
  `usr/sbin/pkg:5913,4890`). **Adopt:** a plan-executor-level `--ask`/`-y`
  toggle rather than per-command confirm flags — cheap and it's the single
  most-repeated pattern in the README's own example section
  (`README.md:322-370`, 6 of 17 examples use `-a`/`--ask`).
- **Pipe-friendly, one-item-per-line output everywhere**, explicitly designed
  for chaining (`README.md:355-359`: `pkg li vim | pkg status -`, `pkg li |
  pkg -a u -` — `-` as "read the package name from stdin"). **Adopt:** the
  CLI's list-producing commands (search, list-installed, list-deps, etc.)
  and its mutating commands (status, uninstall) should compose via stdin the
  same way; this is the single ergonomic idea repeated most across the
  README's examples.
- **`get-only`/`go`: resolve+download without installing** — the closest
  thing Pkg has to a dry-run for the install path (`README.md:207-208,
  343`). Not a true dry-run (files ARE downloaded) but confirms user demand
  for "show me what would happen without committing" — `uw-resolve`'s plan
  being inspectable data *before* any executor runs (proposal §2a: "print
  it, diff it, dry-run it") is a strict improvement over this, worth calling
  out as validated by real demand rather than invented.
- **`func-list`** (`README.md:297`) — the script can enumerate its own
  functions; a small but real precedent for a resolver/CLI that can
  introspect its own command surface, relevant if `cli_args` schema
  generation wants a "list everything this build understands" command.

## 8. Top 10 things `uw-resolve` should take from `Pkg`

1. Model "frozen base" as **layer-membership over currently-mounted SFS
   layers**, not a static manifest — Pkg already ties `devx` freeze status to
   runtime mount state (`usr/sbin/pkg:176-184`); `base/1`/`layer_shadow`
   should be derived the same way, just from a cleaner mount-table source.
2. Add `dependents(Pkg, List)` as a first-class query (`what-needs`,
   `usr/sbin/pkg:6796-6852`) — Puppy users ask this directly, not just as a
   removal-safety side effect.
3. Keep `requested/1` strictly separate from `installed/1` — Pkg's flat
   `user-installed-packages` list conflates them, which is precisely why its
   own `pkg remove` orphan-cleanup code is dead (§3.4); this is the clearest
   place `uw-resolve`'s design is already better, and it's worth stating why.
4. Add an `excluded/1` (blacklist) fact set, orthogonal to `base/1`, consulted
   only in candidate generation — never in removal safety (Pkg accidentally
   couples the two, `usr/sbin/pkg:5825-5828`; don't repeat that).
5. Parse dependency version constraints into `depends/4`'s `Constraint`
   argument instead of discarding them the way `list_deps()` does
   (`usr/sbin/pkg:6160-6162` strips `&ge`/`&lt` rather than keeping them) —
   a straightforward improvement the ingestion spec should make explicitly.
6. Model repo federation as a **priority-ordered fallback chain per repo**
   (`root/.pkg/sources-all`'s trailing fallback-list field), not a flat "all
   repos" bag, for `bleeding-edge`/`repo-scope` candidate ordering.
7. Adopt a small `alias/2` name-normalization fact set
   (`usr/sbin/pkg:227-236`) — cross-repo/cross-distro name drift is real and
   already has a working, if ad hoc, table to start from.
8. `layer_closure`'s output should stay packaging-format-agnostic (Pkg's
   `.pet` vs `.sfs` combine paths share one closure computation, gated only
   at materialization time, `usr/sbin/pkg:3788`) and should keep its
   dependency-before-dependent ordering guarantee even though Pkg's own
   squashfs consumer doesn't strictly need it — safer for sequential
   executors that Pkg itself doesn't have but a future one might.
9. Give the plan executor a universal `--ask` confirm-everything flag and
   design mutating/listing commands to compose over stdin (`pkg li vim | pkg
   status -`) — the most-repeated ergonomic pattern in Pkg's own usage
   examples.
10. Treat `pkg_ldd`'s runtime linkage check as a validated *executor-side
    post-install verification* idea (confirm the plan's effect matches
    reality) without adding it to the resolver spec — real users wanted this
    (it's a top-level command, not hidden), but it needs a filesystem, not a
    fact base.

## Summary: adopt/adapt/already-covered/out-of-scope counts

- **Adopt** (new fact/query shape proposed): §3.3 `excluded/1`, §3.6
  `dependents/2`, §4.1 constraint parsing, §4.2 repo-fallback chains, §4.3
  `alias/2`, §4.4 executor hook points, §5 point 2 (collision-check
  requirement) and point 3 (ordering guarantee) — **8 items**.
- **Adapt** (covered, but a wrinkle ours misses): §3.1 layer-mount-dependent
  `base/1`, §3.2 fixpoint-vs-depth-bound (test-corpus implication), §3.8
  repo-scope/bleeding-edge candidate ordering — **3 items**.
- **Already-covered** (cite proposal section): §3.4 removal safety (§2d
  point 4), §3.5 layer_closure-minus-base (§2d point 3), §6 PPM boundary
  (§5 scope discipline) — **3 items**.
- **Out-of-scope** (say why): §3.7 `deps-check`/`ldd` runtime linkage check
  (needs real binaries on disk, not facts) — **1 item**.

## Contradictions with the current proposal design

None found that require changing `PACKAGE_MANAGER_LOGIC_PROPOSAL.md`'s
architecture. The closest things to a contradiction are both **findings that
uw-resolve is already ahead of Pkg**, not evidence against the design:

- §3.4: Pkg's own orphan-cleanup-on-remove code path is dead (no writer for
  the file it reads) — the `requested/1` vs `installed/1` split the proposal
  already calls for is precisely the fix, not a redundant idea.
- §3.3: Pkg's blacklist accidentally blocks removal of already-installed
  blacklisted packages — a reason to keep `excluded/1` and removal-safety
  logic explicitly decoupled, which the current design already does by not
  conflating them (no predicate currently ties exclusion to removal).

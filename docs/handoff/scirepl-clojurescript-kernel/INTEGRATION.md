# SciREPL ClojureScript kernel — hand-off

Phase 1 of the [ClojureScript-support proposal](https://github.com/s243a/SciREPL/blob/main/docs/proposal-clojurescript-support.md):
a **`clojurescript` kernel** for SciREPL backed by **Scittle/SCI**, the browser
analog of the Lua (Fengari) kernel.

This work targets the **`s243a/SciREPL`** repo, which is outside the UnifyWeaver
agent's push scope — so it's staged here for you to apply. It pairs with the
already-merged UnifyWeaver `clojurescript_target` (phase 2): that target emits
CLJS, and this kernel *runs* it in-app.

Drafted against SciREPL `main` (kernel contract from `www/js/kernels/lua.js`;
integration points per the proposal). Verified: the patch applies cleanly
(`git apply --check` clean against `main`). Not yet run inside SciREPL — see
"Validate" below.

## Contents

| File | Goes to (in SciREPL) |
|------|----------------------|
| `kernels/clojurescript.js` | `www/js/kernels/clojurescript.js` (new file) |
| `integration.patch` | applied at the SciREPL repo root (edits 5 existing files) |

## Apply

```bash
cd /path/to/SciREPL
cp /path/to/this/kernels/clojurescript.js www/js/kernels/clojurescript.js
git apply /path/to/this/integration.patch
node scripts/configure-build.mjs full   # regenerates www/js/kernel_config.js from build-profiles.json
```

## What the patch changes (5 files)

- **`build-profiles.json`** — adds a `clojurescript` language (Scittle CDN
  source, 30s timeout) and enables it in the `light` + `full` profiles. This is
  the source of truth; `www/js/kernel_config.js` is **auto-generated** from it
  (hence the `configure-build.mjs` step — do not hand-edit kernel_config.js).
- **`www/js/file_io.js`** — adds `clojurescript` to `LANGUAGE_META`,
  `IPYNB_KERNELSPEC`, `IPYNB_LANGUAGE_INFO`, and the ipynb-import `knownLangs`
  set (so `.ipynb` round-trips with a real kernelspec, not a python fallback).
- **`www/index.html`** — adds the `CLJS` `#lang-selector` option and the
  `<script src="js/kernels/clojurescript.js">` tag.
- **`www/sw.js`** — precaches the kernel in `APP_SHELL` and bumps
  `CACHE_VERSION` v113 → v114.
- **`www/css/style.css`** — `#lang-selector.clojurescript-active` color (Clojure
  green `#63b132`) and a `.lang-badge.lang-clojurescript` badge.

The kernel file implements the standard contract (`init`, `isReady`, `getName`,
`getLanguage`, `execute`, `getMemoryUsage`, `destroy`, static `displayName`) and
registers via `window.kernelManager.register('clojurescript', ...)`.

## Offline bundling (optional, recommended)

The kernel currently pulls Scittle from jsdelivr. To make it work offline like
brush/typr, vendor it and add a local source:

1. Drop `scittle.js` into `www/vendor/scittle/scittle.js`.
2. In `build-profiles.json`, change the `clojurescript` entry to
   `"runtime": "local"` with a `{ "type": "local", "url": "vendor/scittle/scittle.js" }`
   source (and add it to the `full` profile's `bundle` list), then re-run
   `configure-build.mjs`.
3. Add `./vendor/scittle/scittle.js` to `APP_SHELL` in `www/sw.js`.

## Validate (two spots to check against your Scittle build)

The kernel was drafted by reading the tree, not executed in SciREPL. Confirm:

1. **Scittle global** — `window.scittle.core.eval_string(s)` is the eval entry
   point. If your bundled Scittle exposes a different name, adjust `init()` /
   `execute()`.
2. **stdout capture** — `execute()` binds `cljs.core/*print-fn*` to capture
   `println`/`print` while returning the last form's value (the standard CLJS
   REPL technique). SCI keeps one global context per page, so `def`s and
   namespaces persist across cells. If your build routes prints differently,
   tweak the wrapper string.

Suggested smoke (mirrors `test_prolog_generates_typr.mjs`):

```clojure
(println "hello from scittle")     ; => stdout "hello from scittle"
(defn triple [x] (* x 3)) (triple 5)   ; => result 15, persists to next cell
```

Then the end-to-end `prolog-generates-clojurescript` flow: compile a predicate
with UnifyWeaver's `clojurescript_target`, feed the CLJS to this kernel's
`execute`, assert the output.

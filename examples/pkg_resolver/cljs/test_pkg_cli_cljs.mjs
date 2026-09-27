// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// test_pkg_cli_cljs.mjs -- the `pkg` CLI contract corpus, run against the
// ClojureScript CLI.
//
// This is ../cli/test_pkg_cli.mjs with ONE thing changed: runPkg spawns
// `nbb pkg.cljs` instead of `node pkg.mjs`. Every expectation is the JS lane's
// own -- the SWI-derived generated/expected.json, and renderHuman imported from
// pkg.mjs itself -- so the CLJS CLI is held to exactly the strings the JS CLI
// is held to, assertion for assertion.
//
// Section 6 adds what the import-swap cannot say on its own: for every corpus
// argv, in both spellings, the two CLIs' stdout, stderr and exit code are
// compared BYTE FOR BYTE against each other.
//
//   node --test examples/pkg_resolver/cljs/test_pkg_cli_cljs.mjs

import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { renderHuman, REGISTRY, EXIT_FOR_STATUS } from "../cli/pkg.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CLI_DIR = join(HERE, "..", "cli");
const PKG_CLJS = join(HERE, "pkg.cljs");
const PKG_MJS = join(CLI_DIR, "pkg.mjs");
const GEN = join(CLI_DIR, "generated");
const CLASSPATH = [join(HERE, "..", "..", "cli_args", "cljs"), HERE].join(":");

const EXPECTED = JSON.parse(readFileSync(join(GEN, "expected.json"), "utf8"));

function catalogPath(name) {
  return join(GEN, "catalogs", `${name}.json`);
}

/** The ClojureScript CLI, under nbb. */
function runPkg(argv, env) {
  const r = spawnSync("nbb", ["--classpath", CLASSPATH, PKG_CLJS, ...argv], {
    encoding: "utf8",
    env: { ...process.env, ...(env || {}) }
  });
  if (r.error) throw r.error;
  return { code: r.status, stdout: r.stdout, stderr: r.stderr };
}

/** The JavaScript CLI, for the byte-for-byte cross-check in section 6. */
function runPkgJs(argv, env) {
  const r = spawnSync(process.execPath, [PKG_MJS, ...argv], {
    encoding: "utf8",
    env: { ...process.env, ...(env || {}) }
  });
  if (r.error) throw r.error;
  return { code: r.status, stdout: r.stdout, stderr: r.stderr };
}

// ---------------------------------------------------------------------------
// 1. the derived corpus: JSON form
// ---------------------------------------------------------------------------

test("derived corpus: --json output matches SWI, on every command", async (t) => {
  assert.ok(EXPECTED.cases.length >= 30, "corpus is populated");
  for (const c of EXPECTED.cases) {
    await t.test(`${c.id} --json`, () => {
      const argv = [...c.argv, "--catalog", catalogPath(c.catalog), "--json"];
      const r = runPkg(argv);
      assert.equal(r.stderr, "", `${c.id}: unexpected stderr`);
      const got = JSON.parse(r.stdout);
      assert.deepStrictEqual(got, c.json);
      assert.equal(r.code, c.exit, `${c.id}: exit code`);
    });
  }
});

// ---------------------------------------------------------------------------
// 2. the derived corpus: human form
// ---------------------------------------------------------------------------

test("derived corpus: human output is the SWI document, rendered", async (t) => {
  for (const c of EXPECTED.cases) {
    await t.test(`${c.id} human`, () => {
      const argv = [...c.argv, "--catalog", catalogPath(c.catalog)];
      const r = runPkg(argv);
      assert.equal(r.stderr, "", `${c.id}: unexpected stderr`);
      assert.equal(r.stdout, renderHuman(c.json));
      assert.equal(r.code, c.exit, `${c.id}: exit code`);
    });
  }
});

// ---------------------------------------------------------------------------
// 3. golden text for the cases the demo turns on
// ---------------------------------------------------------------------------

function goldenFor(id) {
  const c = EXPECTED.cases.find((x) => x.id === id);
  assert.ok(c, `case ${id} exists`);
  return runPkg([...c.argv, "--catalog", catalogPath(c.catalog)]);
}

test("golden: a blocked explanation names both ceilings, through an alias", () => {
  const r = goldenFor("f_why_blocked_alias");
  assert.equal(
    r.stdout,
    "firefox-esr is blocked by the frozen base — 2 ceilings\n" +
      "  glibc  needs >=2.35.0  base has 2.31.0\n" +
      "  gtk    needs >=3.24.0  base has 2.24.0\n"
  );
  assert.equal(r.code, 1);
});

test("golden: a coordinated safe-upgrade prints the whole moving set", () => {
  const r = goldenFor("f_safe_upgrade_coord");
  assert.equal(
    r.stdout,
    "glibc -> 2.35.0: coordinated — 3 packages must move together\n" +
      "  glibc  2.35.0\n" +
      "  gtk    3.24.0\n" +
      "  pango  1.50.0\n"
  );
  assert.equal(r.code, 0);
});

test("golden: the audit carries an over-frozen line and a suggestion", () => {
  const r = goldenFor("f_audit");
  assert.equal(
    r.stdout,
    "freeze audit — 5 holds\n" +
      "  busybox  held         modified\n" +
      "  glibc    held         abi_anchor\n" +
      "  gtk      suggest      abi_anchor\n" +
      "  pango    over_frozen\n" +
      "  rxvt     held         footprint\n" +
      "1 over-frozen, 1 with a suggested reason\n"
  );
  assert.equal(r.code, 0);
});

test("golden: an alias resolves inside the resolver (urxvt -> rxvt is a hold)", () => {
  const r = goldenFor("f_safe_upgrade_alias");
  assert.equal(r.stdout, "urxvt -> 9.22.0: safe (cost: footprint)\n");
  assert.equal(r.code, 0);
});

test("golden: a request satisfied by a loaded layer plans to nothing", () => {
  const r = goldenFor("f_install_plan_layer");
  assert.equal(
    r.stdout,
    "plan for gcc — 0 packages (frozen base untouched)\n" +
      "  nothing to install: every request is already satisfied by a loaded layer\n" +
      "install order for gcc:\n" +
      "  (empty)\n"
  );
  assert.equal(r.code, 0);
});

test("golden: classic resolve and the layered plan disagree, and both say so", () => {
  const classic = goldenFor("t_resolve");
  const layered = goldenFor("t_install_plan");
  assert.equal(
    classic.stdout,
    "selection for editor — 3 packages\n" +
      "  editor  1.1.0\n" +
      "  libc    2.0.0\n" +
      "  syntax  2.0.0\n"
  );
  assert.equal(
    layered.stdout,
    "plan for editor — 2 packages (frozen base untouched)\n" +
      "  editor  1.0.0\n" +
      "  syntax  1.0.0\n" +
      "install order for editor:\n" +
      "  1.  syntax  1.0.0\n" +
      "  2.  editor  1.0.0\n"
  );
});

// ---------------------------------------------------------------------------
// 4. usage errors -- every one of these strings is compiled cli_args' own,
//    except the two the CLI itself owns (unknown command / no catalog).
// ---------------------------------------------------------------------------

const TEACHING = () => ["--catalog", catalogPath("teaching")];

test("usage: an unknown command is a CliError with exit 2", () => {
  const r = runPkg(["frobnicate", "x", ...TEACHING()]);
  assert.equal(r.code, 2);
  assert.equal(r.stdout, "");
  assert.match(r.stderr, /^pkg: unknown command: frobnicate\n/);
});

test("usage: a bad flag is the argparser's exact `unknown option` message", () => {
  const r = runPkg(["resolve", "editor", "--nope", ...TEACHING()]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: unknown option --nope\n/);
});

test("usage: a missing positional is the argparser's `missing argument`", () => {
  const r = runPkg(["resolve", ...TEACHING()]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: missing argument: name\n/);
});

test("usage: an extra positional is the argparser's `unexpected extra argument`", () => {
  const r = runPkg(["audit", "extra", ...TEACHING()]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: unexpected extra argument: extra\n/);
});

test("usage: a string option with no value is the argparser's `needs a value`", () => {
  const r = runPkg(["resolve", "editor", "--catalog"]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: --catalog needs a value\n/);
});

test("usage: no catalog at all", () => {
  const r = runPkg(["audit"], { PKG_CATALOG: "" });
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: no catalog: pass --catalog <file> or set PKG_CATALOG\n/);
});

test("usage: an unreadable catalog", () => {
  const r = runPkg(["audit", "--catalog", join(GEN, "nope.json")]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: cannot read catalog .*nope\.json: ENOENT\n/);
});

test("usage: a malformed version", () => {
  const r = runPkg(["safe-upgrade", "libc", "2.0", ...TEACHING()]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: bad version: 2\.0 \(expected MAJOR\.MINOR\.PATCH\)\n/);
});

test("usage: an option before the command is refused, not parsed leniently", () => {
  const r = runPkg([...TEACHING(), "resolve", "editor"]);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /^pkg: the command must come first \(got --catalog\)\n/);
});

// ---------------------------------------------------------------------------
// 5. plumbing
// ---------------------------------------------------------------------------

test("PKG_CATALOG is honoured when --catalog is absent", () => {
  const r = runPkg(["audit"], { PKG_CATALOG: catalogPath("teaching") });
  assert.equal(r.code, 0);
  assert.match(r.stdout, /^freeze audit — 1 hold\n/);
});

test("the registry and the dispatch table do not drift", () => {
  const ALIASES = { layer: "install-plan" };
  const commands = Object.keys(REGISTRY);
  assert.ok(commands.length > 0);
  for (const name of commands) {
    const target = ALIASES[name] || name;
    const r = runPkg([target === "audit" ? "audit" : target, "--catalog", catalogPath("teaching")]);
    assert.doesNotMatch(r.stderr, /unknown command/, `${name} has no handler`);
  }
});

test("every status the CLI can emit has an exit code", () => {
  const seen = new Set(EXPECTED.cases.map((c) => c.json.status));
  for (const s of seen) {
    assert.ok(s in EXIT_FOR_STATUS, `status ${s} has an exit code`);
  }
});

// ---------------------------------------------------------------------------
// 6. the cross-check the import swap cannot make: the two CLIs are
//    indistinguishable on every corpus argv, byte for byte, in both spellings.
// ---------------------------------------------------------------------------

test("the two CLIs agree byte for byte on every corpus argv", async (t) => {
  for (const c of EXPECTED.cases) {
    for (const extra of [[], ["--json"]]) {
      const argv = [...c.argv, "--catalog", catalogPath(c.catalog), ...extra];
      await t.test(`${c.id}${extra.length ? " --json" : ""} cljs == js`, () => {
        const a = runPkg(argv);
        const b = runPkgJs(argv);
        assert.equal(a.stdout, b.stdout, "stdout");
        assert.equal(a.stderr, b.stderr, "stderr");
        assert.equal(a.code, b.code, "exit code");
      });
    }
  }
});

// SPDX-License-Identifier: MIT OR Apache-2.0
//
// cliArgs.wamjs.test.mjs -- peerhailer's own 17-test contract corpus
// (examples/cli_args/oracle/cliArgs.test.mjs), run against the JS-WAM
// compiled parser. The ONLY edit is the import specifier on the line marked
// below; every assertion is the oracle's, verbatim.
//
//   node --test examples/cli_args/wamjs/cliArgs.wamjs.test.mjs
/**
 * The contract matrix from docs/cli-arg-parsing.md, proven at the parser level.
 * (Point 14 — help generated from the same schemas — is a separate feature, not
 * yet built, so it is not asserted here.)
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { parseArgs, CliError } from "./cliArgs.mjs";   // <-- the only edit

const of = (line) => parseArgs(line.split(" ").filter(Boolean));

test("1 & 2: a boolean flag works before or after the positional", () => {
  for (const line of ["block --include-key bob", "block bob --include-key"]) {
    const { positional, flags } = of(line);
    assert.deepEqual(positional, ["block", "bob"], line);
    assert.equal(flags["include-key"], true, line);
  }
});

test("3: an unknown option is refused against the command schema", () => {
  assert.throws(() => of("block bob --include-keey"), CliError);
});

test("4: --include-key=false is off; =anything-else is on", () => {
  assert.equal(of("block bob --include-key=false").flags["include-key"], false);
  assert.equal(of("block bob --include-key=yes").flags["include-key"], true);
});

test("5: a value given to a boolean flag lands as an extra positional and fails", () => {
  assert.throws(() => of("block bob --include-key yes"), /extra argument: yes/);
});

test("6: --state selects the same state before or after the command", () => {
  assert.equal(of("--state P block bob").flags.state, "P");
  assert.equal(of("block bob --state P").flags.state, "P");
});

test("7: `--` preserves a forwarded command with its own flags", () => {
  const { positional } = of("commands add deploy -- ./run.sh --env prod");
  // The handler reads rest.slice(2).join(" "); rest is positional after the command.
  assert.deepEqual(positional.slice(3), ["./run.sh", "--env", "prod"]);
});

test("8: after `--`, a --state token is payload, not the global option", () => {
  const { flags, positional } = of("commands add deploy -- ./run.sh --state child.json");
  assert.equal(flags.state, undefined, "not swallowed as the global");
  assert.ok(positional.includes("--state") && positional.includes("child.json"), "kept as payload");
});

test("9: --key keeps an inline dash-leading PEM as its value", () => {
  const pem = "-----BEGIN-PUBLIC-KEY-----";
  const { flags } = parseArgs(["add", "bob", "--key", pem]);
  assert.equal(flags.key, pem);
});

test("10: --debug is bare-true, spaced, or =valued", () => {
  assert.equal(of("daemon --debug").flags.debug, true);
  assert.equal(of("daemon --debug 2").flags.debug, "2");
  assert.equal(of("daemon --debug=2").flags.debug, "2");
});

test("11: a value handed to a boolean daemon flag fails loudly", () => {
  assert.throws(() => of("daemon --require-target-binding yes"), /extra argument: yes/);
});

test("12: --force is refused on `profiles pin` — it belongs to `profiles remove`", () => {
  assert.throws(() => of("profiles pin trusted --force"), /unknown option --force/);
  // ...and is accepted on remove.
  assert.equal(of("profiles remove temp --force").flags.force, true);
});

test("13: a string option with no value fails", () => {
  assert.throws(() => of("add bob --profile"), /--profile needs a value/);
});

test("a missing required positional fails with the argument name", () => {
  assert.throws(() => of("block"), /missing argument: name/);
});

test("an unmigrated command falls back to the lenient parse (no schema, no error)", () => {
  // `tunnels` has no schema yet: unknown flags are accepted, booleans stay greedy —
  // exactly the legacy behaviour, so the migration can proceed leaf by leaf.
  const { positional, flags } = of("tunnels add acp 127.0.0.1:9100 --anything here");
  assert.equal(positional[0], "tunnels");
  assert.equal(flags.anything, "here");
});

test("regression: daemon accepts --hail-on-encrypted / --hail-on-tls (Fable)", () => {
  assert.equal(of("daemon --hail-on-encrypted tailscale0").flags["hail-on-encrypted"], "tailscale0");
  assert.equal(of("daemon --hail-on-tls eth0").flags["hail-on-tls"], "eth0");
});

test("regression: unblock --key with no name is valid (Fable)", () => {
  const { flags, positional } = of("unblock --key ABCDEF12");
  assert.equal(flags.key, "ABCDEF12");
  assert.deepEqual(positional, ["unblock"], "no name required");
  // ...and the name form still works.
  assert.deepEqual(of("unblock bob").positional, ["unblock", "bob"]);
});

test("the lenient fallback is verbatim: --a --b=c keeps the old greedy reading", () => {
  // An unmigrated command must behave exactly as before: the old parser read
  // `--b=c` (which is not a bare --word) as the value of --a.
  const { flags } = of("tunnels --a --b=c");
  assert.equal(flags.a, "--b=c", "unchanged legacy behaviour on an unmigrated command");
});

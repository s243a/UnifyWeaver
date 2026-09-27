// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// gen_cases.mjs -- the seeded case generator for the differential harness.
//
// Emits argv-lines on stdout, one case per line:
//   1. every line of the 17-test corpus,
//   2. a hand-written "quirk sweep" of oracle edge behaviours (the JS
//      prototype-chain lookups, `--` placements, repeated assignments, case),
//   3. UNIFORM_LINES pseudorandom lines of length 2..7 drawn uniformly from the
//      task's token alphabet,
//   4. ROOTED_LINES more of the same, but with a command as the first token, so
//      the strict path (not just the lenient fallback) is deeply exercised.
//
// The PRNG is mulberry32 with a fixed seed, so the sample is reproducible
// byte-for-byte on any machine.
//
//   node examples/cli_args/gen_cases.mjs > cases.txt

const SEED = 0x1a2b3c4d;
const UNIFORM_LINES = 3000;
const ROOTED_LINES = 2000;
const MIN_LEN = 2;
const MAX_LEN = 7;

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- the token alphabet, exactly as specified for this harness ---------------
const COMMANDS_AND_ACTIONS = [
  "block", "add", "daemon", "unblock", "commands", "route", "profiles",
  "discover", "status", "approve", "send", "pin", "remove", "list",
];
const POSITIONALS = ["bob", "deploy", "./run.sh", "127.0.0.1:9100", "2"];
const OPTIONS = [
  "--include-key", "--include-key=false", "--include-key=yes",
  "--profile", "--profile=x", "--key", "--debug", "--debug=2",
  "--state", "child.json", "--force", "--porfile", "--hail-on-tls", "eth0",
];
const DASH_LEADING = ["-----BEGIN-PUBLIC-KEY-----"];
const TERMINATOR = ["--"];

const ALPHABET = [
  ...COMMANDS_AND_ACTIONS, ...POSITIONALS, ...OPTIONS, ...DASH_LEADING, ...TERMINATOR,
];

// --- 1. the 17-test corpus ---------------------------------------------------
const CORPUS = [
  "block --include-key bob",
  "block bob --include-key",
  "block bob --include-keey",
  "block bob --include-key=false",
  "block bob --include-key=yes",
  "block bob --include-key yes",
  "--state P block bob",
  "block bob --state P",
  "commands add deploy -- ./run.sh --env prod",
  "commands add deploy -- ./run.sh --state child.json",
  "add bob --key -----BEGIN-PUBLIC-KEY-----",
  "daemon --debug",
  "daemon --debug 2",
  "daemon --debug=2",
  "daemon --require-target-binding yes",
  "profiles pin trusted --force",
  "profiles remove temp --force",
  "add bob --profile",
  "block",
  "tunnels add acp 127.0.0.1:9100 --anything here",
  "daemon --hail-on-encrypted tailscale0",
  "daemon --hail-on-tls eth0",
  "unblock --key ABCDEF12",
  "unblock bob",
  "tunnels --a --b=c",
];

// --- 2. the quirk sweep ------------------------------------------------------
// Deliberately probes behaviour the task alphabet cannot reach: JS
// prototype-chain property lookups (`key in GLOBAL_OPTIONS`, `options[key]`,
// `registry[command]`, `entry.actions[action]`), the `__proto__` assignment
// no-op, `--` in every position, repeated assignment order, and the /i flag.
const QUIRKS = [
  "--toString x block bob",
  "block bob --toString v",
  "block bob --constructor",
  "block bob --valueOf",
  "route toString --state P",
  "constructor bob",
  "hasOwnProperty --state P",
  "isPrototypeOf",
  "block bob --propertyIsEnumerable=x",
  "--hasOwnProperty block bob",
  "block bob --__proto__ v",
  "--__proto__ x block bob",
  "tunnels --__proto__=v",
  "profiles valueOf x",
  "commands toString a b c",
  "--state -- block bob",
  "--state --profile block bob",
  "block bob --",
  "tunnels -- bob",
  "-- block bob",
  "--",
  "block bob --state=--x",
  "block --STATE x",
  "block bob --Include-Key",
  "daemon --debug --state P",
  "route send -- --dest x",
  "route send --dest -----BEGIN-PUBLIC-KEY-----",
  "profiles",
  "route",
  "commands list extra",
  "add",
  "add bob address extra",
  "unblock",
  "--name N --state S block bob --state T",
  "--state A block bob --state B --include-key",
  "block bob --state P --name Q",
  "daemon --chat=false --route --ui=0",
  "commands add deploy -- -- ./run.sh",
  "route approve --seal-key -----BEGIN-PUBLIC-KEY----- --port 9",
  "--state",
  "--state --",
  "block --key",
];

const rand = mulberry32(SEED);
const pick = (xs) => xs[Math.floor(rand() * xs.length)];
const pickLen = () => MIN_LEN + Math.floor(rand() * (MAX_LEN - MIN_LEN + 1));

const lines = [...CORPUS, ...QUIRKS];

for (let n = 0; n < UNIFORM_LINES; n += 1) {
  const len = pickLen();
  const tokens = [];
  for (let k = 0; k < len; k += 1) tokens.push(pick(ALPHABET));
  lines.push(tokens.join(" "));
}

for (let n = 0; n < ROOTED_LINES; n += 1) {
  const len = pickLen();
  const tokens = [pick(COMMANDS_AND_ACTIONS)];
  for (let k = 1; k < len; k += 1) tokens.push(pick(ALPHABET));
  lines.push(tokens.join(" "));
}

process.stdout.write(lines.join("\n") + "\n");

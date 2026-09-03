#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
//   node run_probe.mjs <wam-project-dir> <probe.json>
// Must be launched with cwd or imports resolving generated_program.js
// next to resolver_store.mjs inside the wam project dir.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join, resolve } from "node:path";

const wamDir = resolve(process.argv[2]);
const probePath = resolve(process.argv[3]);
const shimUrl = pathToFileURL(join(wamDir, "resolver_store.mjs")).href;
const { resolveLayered } = await import(shimUrl);
const probe = JSON.parse(readFileSync(probePath, "utf8"));
const t0 = process.hrtime.bigint();
const got = resolveLayered(probe.env, probe.args);
const t1 = process.hrtime.bigint();
process.stdout.write("wamjs_result " + JSON.stringify(got) + "\n");
process.stdout.write("wamjs_wall_ms " + (Number(t1 - t0) / 1e6).toFixed(3) + "\n");

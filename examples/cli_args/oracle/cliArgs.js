/**
 * `hail`'s argument grammar, as data — the local typed parser chosen in
 * `docs/cli-arg-parsing.md` (Candidate A). It lives in `src/` rather than
 * `bin/hail.js` so `tsc` checks it and `node --test` can drive it directly.
 *
 * A command with a schema entry is parsed **strictly**: each option is typed, an
 * unknown option is refused, `--` ends option parsing (its tail is preserved as
 * positionals so a forwarded command keeps its own flags), and positional arity is
 * checked. A command with **no** entry falls back to the legacy lenient parse, so
 * unmigrated commands behave exactly as before and the migration can proceed leaf
 * by leaf.
 *
 * The output shape — `{ positional, flags }` — is the one `bin/hail.js` already
 * consumes, so a migrated command's handler reads the *corrected* shape (a boolean
 * is `true`, a positional after a boolean is kept) with no change of its own.
 *
 * @module cliArgs
 */

/** A user-facing parse error; its message is already suitable after a `hail:` prefix. */
export class CliError extends Error {}

/**
 * @typedef {"boolean" | "string" | "optional"} OptionKind
 *   boolean: never takes a value. string: requires one. optional: bare is `true`,
 *   a following non-flag token is its value (used by `--debug [minutes]`).
 *
 * @typedef {{
 *   options?: Record<string, OptionKind>,
 *   positionals?: string[],       // names; a `[name]` is optional, `...name` is variadic (rest)
 *   actions?: Record<string, Omit<CommandSchema, "actions">>, // grouped commands keyed by first positional
 * }} CommandSchema
 */

/** Options accepted by every command, wherever they appear on the line. */
export const GLOBAL_OPTIONS = /** @type {Record<string, OptionKind>} */ ({
  state: "string",
  name: "string",
});

/**
 * Per-command schemas. Only the commands proven against the contract matrix are
 * migrated so far; the rest fall back to the lenient parse. See
 * `docs/cli-arg-parsing.md` for the checklist of what remains.
 *
 * @type {Record<string, CommandSchema>}
 */
export const COMMANDS = {
  block: { positionals: ["name"], options: { "include-key": "boolean" } },
  // name is optional: `unblock --key <pem|fingerprint>` lifts a key with no name.
  unblock: { positionals: ["[name]"], options: { key: "string" } },
  add: {
    positionals: ["name", "[address]"],
    options: { profile: "string", until: "string", key: "string", "key-file": "string", transport: "string" },
  },
  daemon: {
    options: {
      debug: "optional",
      port: "string",
      host: "string",
      "hail-on": "string",
      "hail-on-encrypted": "string",
      "hail-on-tls": "string",
      "tls-cert": "string",
      "tls-key": "string",
      "allow-origin": "string",
      chat: "boolean",
      route: "boolean",
      ui: "boolean",
      "require-target-binding": "boolean",
      "require-sealed": "boolean",
    },
  },
  commands: {
    actions: {
      add: { positionals: ["name", "...command"], options: {} },
      remove: { positionals: ["name"], options: {} },
      list: { positionals: [], options: {} },
    },
  },
  // Live operations on a running daemon's in-memory routed key store (M3b), posted to
  // its control API — so they take a `--control <url>` (default 127.0.0.1:<port>) and the
  // destination identity key via `--dest`/`--dest-file`, not the offline state file.
  route: {
    actions: {
      discover: { positionals: [], options: { dest: "string", "dest-file": "string", control: "string", port: "string" } },
      status: { positionals: [], options: { dest: "string", "dest-file": "string", control: "string", port: "string" } },
      approve: {
        positionals: [],
        options: { dest: "string", "dest-file": "string", "seal-key": "string", "seal-key-file": "string", control: "string", port: "string" },
      },
      send: {
        positionals: ["...message"],
        options: { dest: "string", "dest-file": "string", public: "boolean", ttl: "string", budget: "string", control: "string", port: "string" },
      },
    },
  },
  profiles: {
    actions: {
      add: { positionals: ["name"], options: { allows: "string", description: "string" } },
      remove: { positionals: ["name"], options: { force: "boolean", reassign: "string" } },
      pin: { positionals: ["name"], options: {} },
      unpin: { positionals: ["name"], options: {} },
      list: { positionals: [], options: {} },
    },
  },
};

/** A `--word` or `--word=…` option token (the strict parser's notion). @param {string} token */
const isLongFlag = (token) => /^--[a-z][a-z0-9-]*(=|$)/i.test(token);
/**
 * The *legacy* parser's flag test — a bare `--word` only, so `--x=y` does not read
 * as a flag when it follows another option. Used solely by `parseLenient` to keep
 * unmigrated commands byte-for-byte as they were; the strict parser uses the
 * broader `isLongFlag`.
 * @param {string} token
 */
const looksLikeLegacyFlag = (token) => /^--[a-z][a-z0-9-]*$/i.test(token);

/**
 * Resolve the schema for a command (and, for a grouped command, its action — the
 * first positional). Returns `null` for an unmigrated command, which signals the
 * lenient fallback.
 *
 * @param {string} command
 * @param {string | undefined} maybeAction
 * @param {Record<string, CommandSchema>} registry
 * @returns {{ schema: CommandSchema, actionConsumed: boolean } | null}
 */
function schemaFor(command, maybeAction, registry) {
  const entry = registry[command];
  if (!entry) return null;
  if (entry.actions) {
    const action = maybeAction && entry.actions[maybeAction];
    if (!action) return null; // unknown/absent action → lenient (the handler reports usage)
    return { schema: action, actionConsumed: true };
  }
  return { schema: entry, actionConsumed: false };
}

/**
 * The legacy lenient parse: greedy, untyped, no `--`. Kept verbatim for commands
 * not yet given a schema so their behaviour is unchanged.
 *
 * @param {string[]} argv
 * @returns {{ positional: string[], flags: Record<string, string | boolean> }}
 */
function parseLenient(argv) {
  /** @type {string[]} */
  const positional = [];
  /** @type {Record<string, string | boolean>} */
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === undefined) break;
    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }
    const equals = token.indexOf("=");
    if (equals !== -1) {
      flags[token.slice(2, equals)] = token.slice(equals + 1);
      continue;
    }
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next === undefined || looksLikeLegacyFlag(next)) flags[key] = true;
    else {
      flags[key] = next;
      i += 1;
    }
  }
  return { positional, flags };
}

/**
 * Parse against a resolved leaf schema. `command` (and the action, if any) are
 * already in `positional`; this fills the rest and validates.
 *
 * @param {string[]} argv the full argv
 * @param {number} start index of the first token after command (+action)
 * @param {string[]} positional accumulator, already holding command (+action)
 * @param {CommandSchema} schema
 */
function parseStrict(argv, start, positional, schema) {
  /** @type {Record<string, string | boolean>} */
  const flags = {};
  const options = { ...GLOBAL_OPTIONS, ...(schema.options ?? {}) };
  const values = [];
  let afterTerminator = false;

  for (let i = start; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === undefined) break;
    if (afterTerminator) {
      values.push(token);
      continue;
    }
    if (token === "--") {
      afterTerminator = true; // its tail is payload, never options
      continue;
    }
    if (!token.startsWith("--")) {
      values.push(token);
      continue;
    }
    const equals = token.indexOf("=");
    const key = equals === -1 ? token.slice(2) : token.slice(2, equals);
    const inlineValue = equals === -1 ? null : token.slice(equals + 1);
    const kind = options[key];
    if (!kind) throw new CliError(`unknown option --${key}`);

    if (kind === "boolean") {
      // `--x=false` is off, anything else (including bare) is on.
      flags[key] = inlineValue === null ? true : inlineValue !== "false";
      continue;
    }
    if (inlineValue !== null) {
      flags[key] = inlineValue;
      continue;
    }
    const next = argv[i + 1];
    // A value that is missing or looks like the next option is not a value. Note
    // `isLongFlag` deliberately does not match a `--...` PEM (whose 3rd char is a
    // dash, not a letter), so `--key <inline PEM>` keeps working.
    const value = next !== undefined && next !== "--" && !isLongFlag(next) ? next : null;
    if (kind === "string") {
      if (value === null) throw new CliError(`--${key} needs a value`);
      flags[key] = value;
      i += 1;
      continue;
    }
    // optional: a following value if one is there, else the bare `true`.
    if (value !== null) {
      flags[key] = value;
      i += 1;
    } else {
      flags[key] = true;
    }
  }

  // Positional arity, against the schema's names.
  const names = schema.positionals ?? [];
  const variadic = Boolean(names[names.length - 1]?.startsWith("..."));
  const required = names.filter((n) => !n.startsWith("[") && !n.startsWith("...")).length;
  if (values.length < required) {
    throw new CliError(`missing argument: ${(names[values.length] ?? "argument").replace(/[[\]]/g, "")}`);
  }
  if (!variadic && values.length > names.length) {
    throw new CliError(`unexpected extra argument: ${values[names.length] ?? ""}`);
  }

  positional.push(...values);
  return { positional, flags };
}

/**
 * Parse `hail` arguments. Strict for a scheduled command, lenient otherwise.
 *
 * @param {string[]} argv `process.argv.slice(2)`
 * @param {Record<string, CommandSchema>} [registry]
 * @returns {{ positional: string[], flags: Record<string, string | boolean> }}
 */
export function parseArgs(argv, registry = COMMANDS) {
  // Find the command: the first bare token, skipping any leading global option and
  // its value (globals may appear before the command, e.g. `--state P block bob`).
  let i = 0;
  const leadingGlobals = /** @type {Record<string, string | boolean>} */ ({});
  while (i < argv.length && (argv[i] ?? "").startsWith("--") && argv[i] !== "--") {
    const token = argv[i];
    if (token === undefined) break;
    const equals = token.indexOf("=");
    const key = equals === -1 ? token.slice(2) : token.slice(2, equals);
    if (!(key in GLOBAL_OPTIONS)) break; // a non-global before the command → leave it for the parse below
    if (equals !== -1) {
      leadingGlobals[key] = token.slice(equals + 1);
      i += 1;
    } else {
      const next = argv[i + 1];
      if (next !== undefined && next !== "--" && !isLongFlag(next)) {
        leadingGlobals[key] = next;
        i += 1;
      } else {
        leadingGlobals[key] = true;
      }
      i += 1;
    }
  }

  const command = argv[i];
  if (command === undefined || command.startsWith("--")) return parseLenient(argv);

  const action = argv[i + 1];
  const resolved = schemaFor(command, action, registry);
  if (!resolved) return parseLenient(argv);

  const positional = [command];
  let start = i + 1;
  if (resolved.actionConsumed) {
    positional.push(/** @type {string} */ (action));
    start = i + 2;
  }
  const { positional: pos, flags } = parseStrict(argv, start, positional, resolved.schema);
  return { positional: pos, flags: { ...leadingGlobals, ...flags } };
}

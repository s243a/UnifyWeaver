#!/usr/bin/env ts-node
/**
 * Store and Edit Review Demo
 *
 * Demonstrates:
 * 1. Database-agnostic constraint storage
 * 2. File edit review with constraints
 * 3. (Optional) LLM constraint generation
 *
 * Run with: ts-node store-and-edit-demo.ts
 */

import {
  // Storage
  MemoryConstraintStore,
  SQLiteConstraintStore,
  FileConstraintStore,
  initStore,
  getStore,

  // Constraints
  C,
  validateConstraints,

  // Edit review
  EC,
  registerEditConstraint,
  clearEditConstraints,
  reviewFileEdit,
  computeDiff
} from './index';

// ANSI colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

function section(title: string): void {
  console.log(`\n${CYAN}═══ ${title} ${'═'.repeat(50 - title.length)}${RESET}`);
}

function ok(msg: string): void {
  console.log(`${GREEN}✓${RESET} ${msg}`);
}

function fail(msg: string): void {
  console.log(`${RED}✗${RESET} ${msg}`);
}

function warn(msg: string): void {
  console.log(`${YELLOW}⚠${RESET} ${msg}`);
}

async function demoStorageAbstraction(): Promise<void> {
  section('1. Storage Abstraction Layer');

  console.log('Testing different storage backends...\n');

  // Test Memory Store
  console.log(`${DIM}--- Memory Store ---${RESET}`);
  const memStore = new MemoryConstraintStore();
  await memStore.init();

  const id1 = await memStore.add({
    command: 'test-cmd',
    constraint: C.noShellOps(),
    message: 'Test constraint'
  });
  ok(`Added constraint with ID: ${id1}`);

  const constraints = await memStore.getForCommand('test-cmd');
  ok(`Retrieved ${constraints.length} constraint(s)`);

  await memStore.close();

  // Test SQLite Store
  console.log(`\n${DIM}--- SQLite Store (in-memory) ---${RESET}`);
  const sqlStore = new SQLiteConstraintStore(':memory:');
  await sqlStore.init();

  const ids = await sqlStore.addBatch([
    { command: 'cp', constraint: C.minArgs(2), message: 'cp needs 2 args' },
    { command: 'cp', constraint: C.noShellOps() },
    { command: 'cp', constraint: C.withinSandbox() },
    { command: '*', constraint: C.maxArgs(100), priority: -1 }
  ]);
  ok(`Added ${ids.length} constraints in batch`);

  const cpConstraints = await sqlStore.getForCommand('cp');
  ok(`cp has ${cpConstraints.length} constraints (including global)`);

  const total = await sqlStore.count();
  ok(`Total constraints in store: ${total}`);

  await sqlStore.close();

  // Test File Store
  console.log(`\n${DIM}--- File Store ---${RESET}`);
  const filePath = '/tmp/constraint-demo.json';
  const fileStore = new FileConstraintStore(filePath);
  await fileStore.init();

  await fileStore.add({
    command: 'rm',
    constraint: C.blockedFlags(['-rf', '-fr']),
    message: 'Recursive force delete blocked'
  });
  ok(`Saved constraint to ${filePath}`);

  await fileStore.close();
  ok('File store closed (data persisted)');
}

async function demoGlobalStore(): Promise<void> {
  section('2. Global Store Pattern');

  console.log('Using initStore() for global configuration...\n');

  // Initialize global store with SQLite
  await initStore({ type: 'sqlite', path: ':memory:' });
  const store = getStore();

  await store.add({
    command: 'git',
    constraint: C.argIn(0, ['status', 'log', 'diff', 'branch']),
    message: 'Only read-only git commands allowed'
  });

  ok('Global store initialized with SQLite');

  // Now validateConstraints will use this store
  // (Note: current implementation uses in-memory store in constraints.ts,
  //  this demo shows how the pattern works)
  ok('Store can be swapped at runtime without code changes');
}

async function demoEditReview(): Promise<void> {
  section('3. Edit Review System');

  console.log('Testing constraint-based file edit validation...\n');

  clearEditConstraints();

  // Register constraints for TypeScript files
  registerEditConstraint('**/*.ts', EC.noRemoveSecurityChecks(), {
    message: 'Cannot remove security validation code'
  });

  registerEditConstraint('**/*.ts', EC.maxLinesChanged(50), {
    message: 'Too many lines changed at once'
  });

  registerEditConstraint('**/config.ts', EC.noChangeExports(), {
    message: 'Cannot modify config exports'
  });

  ok('Registered edit constraints for *.ts files');

  // Test case 1: Safe edit
  console.log(`\n${DIM}--- Test: Safe edit ---${RESET}`);
  const original1 = `
function greet(name: string) {
  return 'Hello, ' + name;
}
`;
  const proposed1 = `
function greet(name: string) {
  return \`Hello, \${name}!\`;
}
`;

  const result1 = await reviewFileEdit('src/utils.ts', original1, proposed1);
  if (result1.allowed) {
    ok(`Edit allowed: ${result1.summary}`);
  } else {
    fail(`Edit blocked: ${result1.summary}`);
  }

  // Test case 2: Removing security check
  console.log(`\n${DIM}--- Test: Removing security check ---${RESET}`);
  const original2 = `
function processInput(input: string) {
  validateInput(input);  // Security check
  return transform(input);
}
`;
  const proposed2 = `
function processInput(input: string) {
  return transform(input);
}
`;

  const result2 = await reviewFileEdit('src/process.ts', original2, proposed2);
  if (result2.allowed) {
    fail('Edit should have been blocked!');
  } else {
    ok(`Edit correctly blocked: ${result2.summary}`);
    for (const cr of result2.constraintResults.filter(r => !r.satisfied)) {
      warn(`  Violation: ${cr.message}`);
    }
  }

  // Test case 3: Too many changes
  console.log(`\n${DIM}--- Test: Too many changes ---${RESET}`);
  const original3 = Array(100).fill('// line').join('\n');
  const proposed3 = Array(100).fill('// changed line').join('\n');

  const result3 = await reviewFileEdit('src/big-file.ts', original3, proposed3);
  if (result3.allowed) {
    fail('Edit should have been blocked (too many changes)!');
  } else {
    ok(`Edit correctly blocked: ${result3.summary}`);
  }
}

async function demoDiffAnalysis(): Promise<void> {
  section('4. Diff Analysis');

  const original = `line 1
line 2
line 3
line 4`;

  const proposed = `line 1
modified line 2
line 3
new line 4
line 5`;

  const diff = computeDiff(original, proposed);

  console.log('Diff analysis result:\n');
  for (const d of diff) {
    let icon = ' ';
    let color = RESET;

    switch (d.type) {
      case 'add':
        icon = '+';
        color = GREEN;
        break;
      case 'delete':
        icon = '-';
        color = RED;
        break;
      case 'modify':
        icon = '~';
        color = YELLOW;
        break;
    }

    const content = d.type === 'delete' ? d.originalContent : d.newContent;
    console.log(`${color}${icon} ${d.lineNumber}: ${content}${RESET}`);
  }
}

async function main(): Promise<void> {
  console.log(`${CYAN}Constraint Store & Edit Review Demo${RESET}`);
  console.log('====================================\n');

  await demoStorageAbstraction();
  await demoGlobalStore();
  await demoEditReview();
  await demoDiffAnalysis();

  section('Summary');
  console.log(`
Features demonstrated:
  • Database-agnostic storage (Memory, SQLite, File)
  • Global store pattern for configuration
  • Constraint-based file edit validation
  • Diff analysis for change tracking
  • DSL builders: C (commands), EC (edits)

LLM integration (requires API key):
  • generateConstraints() - Analyze scripts, generate constraints
  • reviewEdit() - LLM-based edit review
  • generateEditConstraints() - Generate edit constraints for files
`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

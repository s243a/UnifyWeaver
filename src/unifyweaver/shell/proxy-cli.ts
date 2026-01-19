#!/usr/bin/env ts-node
/**
 * proxy-cli.ts - CLI wrapper for command-proxy
 *
 * Usage:
 *   ts-node proxy-cli.ts "ls -la"
 *   ts-node proxy-cli.ts --list
 *   ts-node proxy-cli.ts --validate "rm -rf /"
 */

import {
  execute,
  validateCommand,
  parseCommand,
  listCommands,
  Risk
} from './command-proxy';

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help') {
    console.log(`
Command Proxy CLI

Usage:
  proxy-cli.ts <command>           Execute a command through the proxy
  proxy-cli.ts --validate <cmd>    Validate without executing
  proxy-cli.ts --list              List all available commands
  proxy-cli.ts --list-safe         List only safe commands
  proxy-cli.ts --list-json         List commands as JSON
  proxy-cli.ts --help              Show this help

Environment Variables:
  SHELL_ROLE      User role (default: user)
  CONFIRMED       Set to 'true' to confirm high-risk commands
  SANDBOX_ROOT    Root directory for sandbox operations

Examples:
  proxy-cli.ts "ls -la"
  proxy-cli.ts --validate "rm -rf /"
  CONFIRMED=true proxy-cli.ts "rm ./temp-file.txt"
`);
    process.exit(0);
  }

  if (args[0] === '--list') {
    const commands = listCommands();

    console.log('\nAvailable Commands:\n');

    console.log('SAFE:');
    commands
      .filter(c => c.risk === Risk.SAFE)
      .forEach(c => console.log(`  ${c.name.padEnd(12)} - ${c.description}`));

    console.log('\nMODERATE:');
    commands
      .filter(c => c.risk === Risk.MODERATE)
      .forEach(c => console.log(`  ${c.name.padEnd(12)} - ${c.description}`));

    console.log('\nHIGH (requires confirmation):');
    commands
      .filter(c => c.risk === Risk.HIGH)
      .forEach(c => console.log(`  ${c.name.padEnd(12)} - ${c.description}`));

    console.log('\nBLOCKED:');
    commands
      .filter(c => c.risk === Risk.BLOCKED)
      .forEach(c => console.log(`  ${c.name.padEnd(12)} - ${c.description}`));

    process.exit(0);
  }

  if (args[0] === '--list-safe') {
    const commands = listCommands().filter(c => c.risk === Risk.SAFE);
    commands.forEach(c => console.log(c.name));
    process.exit(0);
  }

  if (args[0] === '--list-json') {
    const commands = listCommands();
    console.log(JSON.stringify(commands, null, 2));
    process.exit(0);
  }

  if (args[0] === '--validate') {
    const cmdString = args.slice(1).join(' ');
    const { cmd, args: cmdArgs } = parseCommand(cmdString);
    const result = validateCommand(cmd, cmdArgs, { role: 'user' });

    console.log('\nValidation Result:');
    console.log(`  Command: ${cmd}`);
    console.log(`  Args: ${JSON.stringify(cmdArgs)}`);
    console.log(`  Valid: ${result.ok}`);

    if (!result.ok) {
      console.log(`  Reason: ${result.reason}`);
    }
    if (result.warning) {
      console.log(`  Warning: ${result.warning}`);
    }
    if (result.risk) {
      console.log(`  Risk: ${result.risk}`);
    }
    if (result.requiresConfirmation) {
      console.log(`  Requires Confirmation: yes`);
    }
    if (result.suggestion) {
      console.log(`  Suggestion: ${result.suggestion}`);
    }

    process.exit(result.ok ? 0 : 1);
  }

  // Execute command
  const cmdString = args.join(' ');
  const result = await execute(cmdString, {
    role: process.env.SHELL_ROLE || 'user',
    confirmed: process.env.CONFIRMED === 'true'
  });

  if (!result.success) {
    if (result.requiresConfirmation) {
      console.error(`\n⚠️  This command requires confirmation.`);
      console.error(`   Risk level: ${result.risk}`);
      console.error(`   Description: ${result.description}`);
      if (result.warning) {
        console.error(`   Warning: ${result.warning}`);
      }
      console.error(`\n   To execute, run with CONFIRMED=true`);
      process.exit(2);
    }

    console.error(`\n❌ Command blocked: ${result.error}`);
    if (result.suggestion) {
      console.error(`   ${result.suggestion}`);
    }
    process.exit(1);
  }

  // Output results
  if (result.warning) {
    console.error(`⚠️  Warning: ${result.warning}`);
  }

  if (result.stdout) {
    process.stdout.write(result.stdout);
  }

  if (result.stderr) {
    process.stderr.write(result.stderr);
  }

  process.exit(result.code || 0);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});

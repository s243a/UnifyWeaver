#!/usr/bin/env ts-node
/**
 * LLM Provider Demo
 *
 * Demonstrates available LLM providers (API and CLI).
 *
 * Run with: ts-node llm-provider-demo.ts
 */

import {
  // API providers
  AnthropicProvider,
  OpenAIProvider,

  // CLI providers
  ClaudeCLIProvider,
  GeminiCLIProvider,
  OllamaCLIProvider,
  getAvailableCLIProviders,
  getBestProvider,

  // Types
  LLMProvider
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

function info(msg: string): void {
  console.log(`${DIM}  ${msg}${RESET}`);
}

async function checkAvailability(): Promise<void> {
  section('1. CLI Provider Availability');

  const available = await getAvailableCLIProviders();

  console.log('\nChecking installed CLI tools...\n');

  if (available.includes('claude-cli')) {
    ok('Claude CLI (claude) - Available');
    info('Usage: ClaudeCLIProvider({ model: "sonnet" })');
  } else {
    fail('Claude CLI (claude) - Not found');
    info('Install: npm install -g @anthropic-ai/claude-code');
  }

  if (available.includes('gemini-cli')) {
    ok('Gemini CLI (gemini) - Available');
    info('Usage: GeminiCLIProvider({ model: "gemini-2.0-flash" })');
  } else {
    fail('Gemini CLI (gemini) - Not found');
    info('Install: https://github.com/google-gemini/gemini-cli');
  }

  if (available.includes('ollama')) {
    ok('Ollama (ollama) - Available');
    info('Usage: OllamaCLIProvider({ model: "llama3" })');
  } else {
    fail('Ollama (ollama) - Not found');
    info('Install: https://ollama.ai');
  }

  // Check API keys
  console.log('\nChecking API keys...\n');

  if (process.env.ANTHROPIC_API_KEY) {
    ok('ANTHROPIC_API_KEY - Set');
    info('Usage: AnthropicProvider()');
  } else {
    fail('ANTHROPIC_API_KEY - Not set');
  }

  if (process.env.OPENAI_API_KEY) {
    ok('OPENAI_API_KEY - Set');
    info('Usage: OpenAIProvider()');
  } else {
    fail('OPENAI_API_KEY - Not set');
  }
}

async function demoBestProvider(): Promise<void> {
  section('2. Auto-Select Best Provider');

  try {
    const provider = await getBestProvider();
    ok(`Best provider: ${provider.name}`);
  } catch (err) {
    fail(`No provider available: ${err instanceof Error ? err.message : String(err)}`);
  }
}

async function demoProviderUsage(): Promise<void> {
  section('3. Provider Usage Examples');

  console.log(`
${CYAN}API Providers:${RESET}

  // Anthropic (requires ANTHROPIC_API_KEY)
  const claude = new AnthropicProvider({ model: 'claude-sonnet-4-20250514' });
  const response = await claude.complete('Hello!');

  // OpenAI-compatible (works with local servers too)
  const openai = new OpenAIProvider({
    model: 'gpt-4',
    baseUrl: 'http://localhost:1234/v1'  // For local models
  });

${CYAN}CLI Providers:${RESET}

  // Claude Code CLI
  const claudeCli = new ClaudeCLIProvider({ model: 'sonnet' });
  const response = await claudeCli.complete('Hello!');

  // Gemini CLI
  const gemini = new GeminiCLIProvider({ model: 'gemini-2.0-flash' });

  // Ollama (local models)
  const ollama = new OllamaCLIProvider({ model: 'llama3' });

${CYAN}Auto-detection:${RESET}

  // Get best available provider
  const provider = await getBestProvider();

  // Check what's available
  const available = await getAvailableCLIProviders();
  // Returns: ['claude-cli', 'gemini-cli', 'ollama'] (those that are installed)

${CYAN}Constraint Generation:${RESET}

  import { generateConstraints } from './index';

  // Uses provider for constraint generation
  const result = await generateConstraints('./script.sh', content, {
    provider: new ClaudeCLIProvider({ model: 'sonnet' })
  });
`);
}

async function testProvider(provider: LLMProvider, name: string): Promise<void> {
  console.log(`\n${DIM}Testing ${name}...${RESET}`);

  try {
    const response = await provider.complete(
      'Respond with exactly: "Hello from LLM"',
      { maxTokens: 50 }
    );

    if (response.toLowerCase().includes('hello')) {
      ok(`${name}: ${response.slice(0, 50)}`);
    } else {
      fail(`${name}: Unexpected response`);
      info(response.slice(0, 100));
    }
  } catch (err) {
    fail(`${name}: ${err instanceof Error ? err.message : String(err)}`);
  }
}

async function runLiveTests(): Promise<void> {
  section('4. Live Provider Tests (Optional)');

  const args = process.argv.slice(2);

  if (!args.includes('--test')) {
    console.log(`\n${DIM}Skipping live tests. Run with --test to enable.${RESET}`);
    console.log(`${DIM}Example: ts-node llm-provider-demo.ts --test${RESET}`);
    return;
  }

  console.log('\nRunning live tests...');

  const available = await getAvailableCLIProviders();

  if (available.includes('claude-cli')) {
    await testProvider(new ClaudeCLIProvider(), 'Claude CLI');
  }

  if (available.includes('gemini-cli')) {
    await testProvider(new GeminiCLIProvider(), 'Gemini CLI');
  }

  if (available.includes('ollama')) {
    await testProvider(new OllamaCLIProvider({ model: 'llama3' }), 'Ollama');
  }

  if (process.env.ANTHROPIC_API_KEY) {
    await testProvider(new AnthropicProvider(), 'Anthropic API');
  }

  if (process.env.OPENAI_API_KEY) {
    await testProvider(new OpenAIProvider(), 'OpenAI API');
  }
}

async function main(): Promise<void> {
  console.log(`${CYAN}LLM Provider Demo${RESET}`);
  console.log('=================\n');

  await checkAvailability();
  await demoBestProvider();
  await demoProviderUsage();
  await runLiveTests();

  section('Summary');
  console.log(`
Available provider types:
  ${CYAN}API-based:${RESET}     AnthropicProvider, OpenAIProvider
  ${CYAN}CLI-based:${RESET}     ClaudeCLIProvider, GeminiCLIProvider, OllamaCLIProvider
  ${CYAN}Custom:${RESET}        GenericCLIProvider (for any CLI tool)

Auto-detection:
  getBestProvider()         - Returns best available provider
  getAvailableCLIProviders() - Lists installed CLI tools
`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

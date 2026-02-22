# uwsal — UnifyWeaver Scripted Agent Loop

A hand-crafted Python agent loop with multi-backend LLM support.
Originally developed as part of [UnifyWeaver](https://github.com/s243a/UnifyWeaver),
this code serves as the reference implementation that the Prolog generator
(`agent_loop_module.pl`) was built to reproduce.

## Backends

- **coro**: Coro-code CLI backend using single-task mode
- **claude_code**: Claude Code CLI backend using print mode
- **gemini**: Gemini CLI backend
- **claude_api**: Anthropic Claude API backend
- **openai_api**: OpenAI API backend
- **ollama_api**: Ollama REST API backend for local models
- **ollama_cli**: Ollama CLI backend using 'ollama run' command
- **openrouter_api**: OpenRouter API backend with model routing

## Tools

- **bash**: Execute a bash command
- **read**: Read a file
- **write**: Write content to a file
- **edit**: Edit a file with search/replace

## Usage

```bash
python3 agent_loop.py              # interactive
python3 agent_loop.py "prompt"     # single prompt
python3 agent_loop.py -b claude    # use Claude API
```

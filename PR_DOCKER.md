# chore: Add Docker Development Environment

## Summary
This PR adds a standard `docker` directory to the repository containing a `Dockerfile` and helper scripts. This environment is pre-configured with all dependencies needed to build and run UnifyWeaver targets (Python, C#, Go) and the new Semantic Runtime.

## Included Tools
- **Core**: SWI-Prolog, Node.js
- **Python**: Python 3, `lxml`, `numpy`, `onnxruntime`, `sqlite3`
- **Go**: Golang (latest via apt)
- **.NET**: .NET SDK 8.0, PowerShell
- **Utils**: `jq`, `ripgrep`, `xmlstarlet`, `sqlite3`

## Usage
```bash
cd docker
./docker-dev.sh
```
This drops you into a bash shell with all tools available.

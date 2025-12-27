# RPyC Integration Example

This example demonstrates using RPyC for network-based RPC communication between Prolog and Python through UnifyWeaver's glue system.

## Overview

RPyC (Remote Python Call) provides live object proxies over the network, filling the gap between:
- **Pipe transport**: Serialized data, same machine
- **HTTP transport**: Serialized data, network
- **Janus**: Live objects, in-process only
- **RPyC**: Live objects, network-capable

## Quick Start

### 1. Install RPyC

```bash
pip install rpyc
```

### 2. Start the Server

```bash
python rpyc_server.py
```

You should see:
```
UnifyWeaver RPyC Demo Server
Listening on port 18812
Press Ctrl+C to stop
```

### 3. Run the Demo

In a new terminal:

```bash
swipl rpyc_demo.pl
```

```prolog
?- run_demo.
```

## Demo Features

1. **Code Generation** - Generate client wrappers and service code
2. **Connection Test** - Connect to RPyC server
3. **Remote Computation** - Execute math and code remotely
4. **Proxy Layers** - Explore the four proxy layers
5. **Transport Comparison** - Understand when to use RPyC

## Proxy Layers

The RPyC transport provides four proxy layers:

| Layer | Property | Use Case |
|-------|----------|----------|
| `root` | Direct exposed_ access | Simple RPC calls |
| `wrapped_root` | Safe attribute access | Prevent accidental execution |
| `auto_root` | Automatic wrapping | General use |
| `smart_root` | Local-class-aware | Code generation |

## Security Modes

- **unsecured**: Development only (shown in this demo)
- **ssh**: SSH tunnel (recommended for production)
- **ssl**: SSL/TLS with certificates

For production use, always use `ssh` or `ssl` modes.

## Files

- `rpyc_demo.pl` - Prolog demo script
- `rpyc_server.py` - Python RPyC server

## Related Documentation

- [RPYC Transport Proposal](../../docs/proposals/RPYC_TRANSPORT_PROPOSAL.md)
- [Chapter 21: Janus Integration](../../education/book-07-cross-target-glue/21_janus_integration.md)

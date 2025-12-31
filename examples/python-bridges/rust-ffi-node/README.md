# Node.js + React + Rust FFI + RPyC Example

Full-stack TypeScript application demonstrating Python access from Node.js via the Rust FFI bridge.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (React + TypeScript)                                │
│   - UI for data input and visualization                     │
│   - Calls Express API via fetch                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (port 3001)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Node.js Backend (Express + TypeScript)                      │
│   - REST API endpoints                                      │
│   - Uses koffi to call Rust library                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ FFI (librpyc_bridge.so)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Rust cdylib (PyO3)                                          │
│   - Embeds CPython                                          │
│   - Manages RPyC connection                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ TCP (port 18812)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ RPyC Server (Python)                                        │
│   - NumPy, SciPy, pandas, scikit-learn                      │
│   - Custom ML services                                      │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Node.js 18+**
   ```bash
   node --version  # Should be 18+
   ```

2. **Rust toolchain** (for building the FFI library)
   ```bash
   cargo --version
   ```

3. **Python 3.8+ with RPyC and NumPy**
   ```bash
   pip install rpyc numpy
   ```

4. **Build the Rust FFI library**
   ```bash
   cd ../rust-ffi-go
   cargo build --release
   cp target/release/librpyc_bridge.so ../rust-ffi-node/
   ```

## Quick Start

### 1. Start RPyC Server

```bash
python ../../rpyc-integration/rpyc_server.py
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run Tests

```bash
npm run test
```

Expected output:
```
Node.js + Rust FFI + RPyC Integration Test
==========================================

Connecting to RPyC server...
  Connected!

Test 1: math.sqrt(16)
  Result: 4
  PASSED!

Test 2: numpy.mean([1, 2, 3, 4, 5])
  Result: 3
  PASSED!

Test 3: math.pi
  Result: 3.141592653589793
  PASSED!

==========================================
All tests passed!
```

### 4. Start the API Server

```bash
npm run dev
```

Server runs at http://localhost:3001

### 5. Test API Endpoints

```bash
# Health check
curl http://localhost:3001/health

# Calculate mean
curl -X POST http://localhost:3001/numpy/mean \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3, 4, 5]}'

# Square root
curl -X POST http://localhost:3001/math/sqrt \
  -H "Content-Type: application/json" \
  -d '{"value": 16}'

# Get pi
curl http://localhost:3001/math/pi

# Generic Python call
curl -X POST http://localhost:3001/python/call \
  -H "Content-Type: application/json" \
  -d '{"module": "numpy", "func": "sum", "args": [[1, 2, 3, 4, 5]]}'
```

## API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/health` | GET | - | Health check |
| `/connect` | POST | - | Connect to RPyC server |
| `/disconnect` | POST | - | Disconnect |
| `/python/call` | POST | `{module, func, args}` | Generic Python call |
| `/python/attr/:m/:a` | GET | - | Get module attribute |
| `/numpy/mean` | POST | `{data: number[]}` | Calculate mean |
| `/numpy/std` | POST | `{data: number[]}` | Calculate std dev |
| `/math/sqrt` | POST | `{value: number}` | Square root |
| `/math/pi` | GET | - | Get pi constant |

## React Frontend

See `frontend/` directory for the React UI.

```bash
cd frontend
npm install
npm start
```

Opens at http://localhost:3000

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3001 | API server port |
| `RPYC_HOST` | localhost | RPyC server host |
| `RPYC_PORT` | 18812 | RPyC server port |

## Project Structure

```
rust-ffi-node/
├── package.json
├── tsconfig.json
├── src/
│   ├── rpyc_bridge.ts    # FFI wrapper
│   ├── server.ts         # Express API
│   └── test.ts           # Test script
├── frontend/             # React app
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   └── ...
│   └── public/
└── librpyc_bridge.so     # Copied from rust-ffi-go
```

## Troubleshooting

### "Failed to connect to RPyC server"

Make sure RPyC server is running:
```bash
python ../../rpyc-integration/rpyc_server.py
```

### "librpyc_bridge.so not found"

Build and copy the Rust library:
```bash
cd ../rust-ffi-go
cargo build --release
cp target/release/librpyc_bridge.so ../rust-ffi-node/
```

# Network Glue Examples

Examples demonstrating UnifyWeaver's network communication capabilities.

## Distributed Processing Pipeline

A microservices-based data processing system with four services:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Distributed Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Client                                                             │
│     │                                                                │
│     ▼                                                                │
│  ┌─────────────┐                                                    │
│  │   Gateway   │ (Go, :8080)                                        │
│  │   Router    │                                                    │
│  └──────┬──────┘                                                    │
│         │                                                            │
│    ┌────┴────┬─────────────┐                                        │
│    ▼         ▼             ▼                                        │
│ ┌──────┐ ┌──────┐    ┌──────────┐                                  │
│ │Trans-│ │  ML  │    │Aggregator│                                  │
│ │form  │ │Svc   │    │  (Rust)  │                                  │
│ │(Py)  │ │(Go)  │    │  :8083   │                                  │
│ │:8081 │ │:8082 │    └──────────┘                                  │
│ └──────┘ └──────┘                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Language | Port | Purpose |
|---------|----------|------|---------|
| Gateway | Go | 8080 | API entry point, routing |
| Transform | Python | 8081 | Data normalization |
| ML Service | Go | 8082 | Prediction, classification |
| Aggregator | Rust | 8083 | High-performance stats |

### Files

| File | Description |
|------|-------------|
| `distributed_pipeline.pl` | Service definitions and code generator |
| Generated outputs: |
| `gateway.go` | Go API gateway |
| `transform_service.py` | Python transform service |
| `ml_service.go` | Go ML service |
| `aggregator_service.rs` | Rust aggregator |
| `client.py` | Python client library |
| `docker-compose.yml` | Container orchestration |

### Usage

```bash
# Generate all service code
swipl distributed_pipeline.pl

# Option 1: Run with Docker
docker-compose up

# Option 2: Run locally
go run gateway.go &
python transform_service.py &
go run ml_service.go &
# Note: Rust service requires cargo build first
```

### API Endpoints

**Gateway (port 8080)**
```
POST /api/process   - Process single record
POST /api/batch     - Process multiple records
GET  /health        - Health check
```

**Transform (port 8081)**
```
POST /transform     - Transform single record
POST /batch         - Transform batch
GET  /health        - Health check
```

**ML Service (port 8082)**
```
POST /predict       - Generate prediction
POST /classify      - Classify record
GET  /health        - Health check
```

**Aggregator (port 8083)**
```
POST /aggregate     - Aggregate results
POST /stats         - Compute statistics
GET  /health        - Health check
```

### Example Request

```bash
# Process a single record
curl -X POST http://localhost:8080/api/process \
  -H "Content-Type: application/json" \
  -d '{"data": {"id": "001", "value": 75, "timestamp": "2025-01-15T10:30:00Z"}}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "001",
    "value": 75,
    "timestamp": "2025-01-15T10:30:00Z",
    "normalized": 0.75,
    "hour": 10,
    "day_of_week": 2,
    "transformed": true,
    "prediction": {
      "score": 0.628,
      "confidence": 0.912,
      "label": "medium"
    }
  }
}
```

### Batch Processing

```bash
# Process multiple records
curl -X POST http://localhost:8080/api/batch \
  -H "Content-Type: application/json" \
  -d '{"data": [
    {"id": "001", "value": 25},
    {"id": "002", "value": 75},
    {"id": "003", "value": 90}
  ]}'
```

### Aggregation

```bash
# First, get batch results
results=$(curl -s -X POST http://localhost:8080/api/batch \
  -H "Content-Type: application/json" \
  -d '{"data": [{"value": 25}, {"value": 50}, {"value": 75}]}')

# Then aggregate
curl -X POST http://localhost:8083/stats \
  -H "Content-Type: application/json" \
  -d "{\"data\": $results}"
```

**Stats Response:**
```json
{
  "success": true,
  "data": {
    "count": 3,
    "sum": 1.524,
    "mean": 0.508,
    "min": 0.312,
    "max": 0.691,
    "std_dev": 0.155
  }
}
```

### Service Communication Pattern

```
1. Client → Gateway
   POST /api/process {data: {...}}

2. Gateway → Transform
   POST /transform {data: {...}}
   ← {normalized: 0.75, ...}

3. Gateway → ML Service
   POST /predict {data: {...}}
   ← {prediction: {score: 0.6, ...}}

4. Gateway → Client
   ← {success: true, data: {...}}
```

### Python Client Usage

```python
from client import gateway_api_process, gateway_api_batch

# Single record
result = gateway_api_process({"id": "001", "value": 50})
print(result)

# Batch
results = gateway_api_batch([
    {"id": "001", "value": 25},
    {"id": "002", "value": 75},
])
print(results)
```

### Health Checks

```bash
# Check all services
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8082/health
curl http://localhost:8083/health
```

### Error Handling

All services return consistent error format:

```json
{
  "success": false,
  "error": "Error description"
}
```

HTTP status codes:
- 200: Success
- 400: Bad request (invalid input)
- 500: Internal server error

### Scaling Considerations

For production deployment:

1. **Load Balancing**: Add nginx/HAProxy in front of gateway
2. **Service Discovery**: Use Consul/etcd instead of hardcoded URLs
3. **Circuit Breaker**: Add resilience patterns for service calls
4. **Monitoring**: Add Prometheus metrics and Grafana dashboards
5. **Logging**: Centralize logs with ELK or similar

### Dependencies

**Gateway & ML Service (Go):**
- net/http (standard library)

**Transform Service (Python):**
```bash
pip install flask flask-cors
```

**Aggregator (Rust):**
```toml
[dependencies]
actix-web = "4"
actix-cors = "0.6"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Client (Python):**
```bash
pip install requests
```

# KG Topology Docker Cluster

Docker Compose configuration for running a local KG Topology semantic search cluster.

## Quick Start

```bash
# Start the cluster
docker-compose up -d

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8081/kg/health
curl http://localhost:8082/kg/health
curl http://localhost:8083/kg/health

# Stop the cluster
docker-compose down
```

## Components

| Service | Port | Description |
|---------|------|-------------|
| consul | 8500 | Service discovery (UI at http://localhost:8500) |
| kg-node-csv | 8081 | CSV/delimited data expert |
| kg-node-json | 8082 | JSON/API expert |
| kg-node-sql | 8083 | SQL/database expert |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards (admin/admin) |

## Endpoints

Each KG node exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/kg/health` | GET | Health check |
| `/kg/metrics` | GET | Prometheus metrics |
| `/kg/query` | POST | Semantic search |
| `/kg/federated` | POST | Federated query |
| `/kg/federation/stats` | GET | Federation statistics |

## Example Queries

### Local Query

```bash
curl -X POST http://localhost:8081/kg/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "How do I parse CSV files?", "top_k": 5}'
```

### Federated Query

```bash
curl -X POST http://localhost:8081/kg/federated \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "How do I serialize data?",
    "top_k": 5,
    "federation_k": 3,
    "strategy": "diversity"
  }'
```

## Monitoring

- **Consul UI**: http://localhost:8500
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

The Grafana dashboard "KG Topology Cluster" shows:
- Healthy nodes count
- Queries per second by node
- Average query latency
- Shortcut hit rate (path folding effectiveness)
- Federated query rate

## Adding Data

Data must be loaded into each node's knowledge base. Mount volumes:

```yaml
volumes:
  - ./my-csv-data:/app/data
  - ./my-csv-embeddings:/app/embeddings
```

Or use the KG API to add Q/A pairs:

```python
from kg_topology_api import DistributedKGTopologyAPI

api = DistributedKGTopologyAPI(db_path="/app/data/kg.db")
api.add_qa_pair(
    question="How do I parse CSV?",
    answer="Use csv.reader() in Python...",
    model_name="all-MiniLM-L6-v2"
)
```

## Configuration

Environment variables for each node:

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ID` | kg-node | Unique node identifier |
| `NODE_PORT` | 8080 | Server port |
| `CONSUL_HOST` | localhost | Consul host |
| `CONSUL_PORT` | 8500 | Consul port |
| `INTERFACE_TOPICS` | general | Comma-separated topics |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |

## Scaling

To add more nodes:

```yaml
services:
  kg-node-python:
    build:
      context: ../..
      dockerfile: docker/kg-topology/Dockerfile.kg-node
    environment:
      - NODE_ID=python-node
      - NODE_PORT=8084
      - INTERFACE_TOPICS=python,scripting,automation
      # ...
    ports:
      - "8084:8084"
```

## Troubleshooting

### Nodes not registering with Consul

```bash
# Check Consul logs
docker logs kg-consul

# Verify node can reach Consul
docker exec kg-node-csv curl http://consul:8500/v1/status/leader
```

### No data in Grafana

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify nodes are exposing metrics: `curl http://localhost:8081/kg/metrics`
3. Ensure Grafana datasource is configured

### High latency

1. Check if embeddings are cached
2. Verify sufficient memory allocation
3. Consider enabling path folding for repeated queries

## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Main compose configuration |
| `Dockerfile.kg-node` | KG node container image |
| `entrypoint.sh` | Node startup script |
| `kg_node_server.py` | Flask server for KG node |
| `prometheus.yml` | Prometheus scrape configuration |
| `grafana/` | Grafana dashboards and datasources |

## Related Documentation

- [KG Production Deployment Guide](../../docs/guides/KG_PRODUCTION_DEPLOYMENT.md)
- [KG Topology Roadmap](../../docs/proposals/ROADMAP_KG_TOPOLOGY.md)
- [Federated Query Algebra](../../docs/proposals/FEDERATED_QUERY_ALGEBRA.md)

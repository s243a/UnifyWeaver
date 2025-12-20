# KG Topology Production Deployment Guide

This guide covers deploying a distributed semantic search cluster using UnifyWeaver's KG Topology system.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Consul (for service discovery)
- Basic understanding of KG Topology (see `docs/proposals/ROADMAP_KG_TOPOLOGY.md`)

## Architecture Overview

A production KG topology deployment consists of:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
│                    (HAProxy / Nginx / AWS ALB)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │  KG Node  │   │  KG Node  │   │  KG Node  │
        │  (CSV)    │   │  (JSON)   │   │  (SQL)    │
        │ port:8081 │   │ port:8082 │   │ port:8083 │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────┐
                    │     Consul      │
                    │  (Discovery)    │
                    │   port:8500     │
                    └─────────────────┘
```

## Quick Start: Local Development Cluster

### Step 1: Start Consul

```bash
# Run Consul in development mode
docker run -d --name consul \
  -p 8500:8500 \
  -p 8600:8600/udp \
  consul:1.15 agent -dev -client=0.0.0.0 -ui
```

Verify Consul is running: http://localhost:8500

### Step 2: Run the Docker Compose Cluster

```bash
# From project root
cd docker/kg-topology

# Build and start the cluster
docker-compose up -d

# View logs
docker-compose logs -f

# Check node status
curl http://localhost:8081/kg/health
curl http://localhost:8082/kg/health
curl http://localhost:8083/kg/health
```

### Step 3: Verify Federation

```bash
# Query the federation endpoint
curl -X POST http://localhost:8081/kg/federated \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "How do I parse CSV files?",
    "top_k": 5,
    "federation_k": 3
  }'
```

## Docker Compose Configuration

Create `docker/kg-topology/docker-compose.yml`:

```yaml
version: '3.8'

services:
  consul:
    image: consul:1.15
    container_name: kg-consul
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    command: agent -server -bootstrap-expect=1 -ui -client=0.0.0.0
    healthcheck:
      test: ["CMD", "consul", "info"]
      interval: 10s
      timeout: 5s
      retries: 3

  kg-node-csv:
    build:
      context: ../..
      dockerfile: docker/kg-topology/Dockerfile.kg-node
    container_name: kg-node-csv
    environment:
      - NODE_ID=csv-node
      - NODE_PORT=8081
      - INTERFACE_TOPICS=csv,delimited,tabular
      - CONSUL_HOST=consul
      - CONSUL_PORT=8500
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - PYTHONUNBUFFERED=1
    ports:
      - "8081:8081"
    depends_on:
      consul:
        condition: service_healthy
    volumes:
      - csv-data:/app/data
      - csv-embeddings:/app/embeddings
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8081/kg/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  kg-node-json:
    build:
      context: ../..
      dockerfile: docker/kg-topology/Dockerfile.kg-node
    container_name: kg-node-json
    environment:
      - NODE_ID=json-node
      - NODE_PORT=8082
      - INTERFACE_TOPICS=json,serialization,api
      - CONSUL_HOST=consul
      - CONSUL_PORT=8500
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - PYTHONUNBUFFERED=1
    ports:
      - "8082:8082"
    depends_on:
      consul:
        condition: service_healthy
    volumes:
      - json-data:/app/data
      - json-embeddings:/app/embeddings
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8082/kg/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  kg-node-sql:
    build:
      context: ../..
      dockerfile: docker/kg-topology/Dockerfile.kg-node
    container_name: kg-node-sql
    environment:
      - NODE_ID=sql-node
      - NODE_PORT=8083
      - INTERFACE_TOPICS=sql,database,queries
      - CONSUL_HOST=consul
      - CONSUL_PORT=8500
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - PYTHONUNBUFFERED=1
    ports:
      - "8083:8083"
    depends_on:
      consul:
        condition: service_healthy
    volumes:
      - sql-data:/app/data
      - sql-embeddings:/app/embeddings
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8083/kg/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: kg-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:10.0.0
    container_name: kg-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

volumes:
  csv-data:
  csv-embeddings:
  json-data:
  json-embeddings:
  sql-data:
  sql-embeddings:
  prometheus-data:
  grafana-data:
```

## KG Node Dockerfile

Create `docker/kg-topology/Dockerfile.kg-node`:

```dockerfile
# KG Topology Node
# Semantic search node with Kleinberg routing and federation

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-kg.txt .
RUN pip install --no-cache-dir -r requirements-kg.txt

# Copy KG topology runtime
COPY src/unifyweaver/targets/python_runtime/*.py ./

# Copy entrypoint script
COPY docker/kg-topology/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:${NODE_PORT:-8080}/kg/health || exit 1

# Expose the node port (will be overridden by environment)
EXPOSE 8080

# Run the KG node
ENTRYPOINT ["./entrypoint.sh"]
```

## Entrypoint Script

Create `docker/kg-topology/entrypoint.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Configuration from environment
NODE_ID="${NODE_ID:-kg-node}"
NODE_PORT="${NODE_PORT:-8080}"
CONSUL_HOST="${CONSUL_HOST:-localhost}"
CONSUL_PORT="${CONSUL_PORT:-8500}"
INTERFACE_TOPICS="${INTERFACE_TOPICS:-general}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

echo "Starting KG Node: ${NODE_ID}"
echo "  Port: ${NODE_PORT}"
echo "  Consul: ${CONSUL_HOST}:${CONSUL_PORT}"
echo "  Topics: ${INTERFACE_TOPICS}"

# Wait for Consul to be available
echo "Waiting for Consul..."
until curl -sf "http://${CONSUL_HOST}:${CONSUL_PORT}/v1/status/leader" > /dev/null; do
    echo "  Consul not ready, waiting..."
    sleep 2
done
echo "Consul is ready."

# Start the KG node
exec python3 kg_node_server.py \
    --node-id "${NODE_ID}" \
    --port "${NODE_PORT}" \
    --consul-host "${CONSUL_HOST}" \
    --consul-port "${CONSUL_PORT}" \
    --topics "${INTERFACE_TOPICS}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --db-path "/app/data/kg.db" \
    --embeddings-path "/app/embeddings"
```

## Consul Health Checks

KG nodes automatically register with Consul including health checks. The registration payload:

```json
{
  "ID": "kg-csv-node-1",
  "Name": "kg_node",
  "Tags": ["kg_node", "csv_expert"],
  "Address": "192.168.1.10",
  "Port": 8081,
  "Meta": {
    "semantic_centroid": "base64_encoded_vector...",
    "embedding_model": "all-MiniLM-L6-v2",
    "interface_topics": "[\"csv\", \"delimited\", \"tabular\"]",
    "corpus_id": "csv_docs_v1"
  },
  "Check": {
    "HTTP": "http://192.168.1.10:8081/kg/health",
    "Interval": "30s",
    "Timeout": "5s",
    "DeregisterCriticalServiceAfter": "5m"
  }
}
```

### Manual Health Check Registration

```bash
# Register health check via Consul API
curl -X PUT http://localhost:8500/v1/agent/check/register \
  -H "Content-Type: application/json" \
  -d '{
    "ID": "kg-csv-node-health",
    "Name": "KG CSV Node Health",
    "ServiceID": "kg-csv-node-1",
    "HTTP": "http://kg-node-csv:8081/kg/health",
    "Interval": "30s",
    "Timeout": "5s"
  }'
```

## Prometheus Configuration

Create `docker/kg-topology/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'consul'
    static_configs:
      - targets: ['consul:8500']

  - job_name: 'kg-nodes'
    consul_sd_configs:
      - server: 'consul:8500'
        services:
          - 'kg_node'
    relabel_configs:
      - source_labels: [__meta_consul_tags]
        regex: '.*,kg_node,.*'
        action: keep
      - source_labels: [__meta_consul_service_metadata_interface_topics]
        target_label: topics
      - source_labels: [__meta_consul_service]
        target_label: service

  - job_name: 'kg-nodes-metrics'
    consul_sd_configs:
      - server: 'consul:8500'
        services:
          - 'kg_node'
    metrics_path: '/kg/metrics'
    relabel_configs:
      - source_labels: [__meta_consul_tags]
        regex: '.*,kg_node,.*'
        action: keep
```

## Grafana Dashboard

Create `docker/kg-topology/grafana/dashboards/kg-topology.json`:

```json
{
  "dashboard": {
    "title": "KG Topology Cluster",
    "panels": [
      {
        "title": "Query Latency (p99)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(kg_query_latency_seconds_bucket[5m]))",
            "legendFormat": "{{node_id}}"
          }
        ]
      },
      {
        "title": "Federation Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "sum(rate(kg_federated_queries_success[5m])) / sum(rate(kg_federated_queries_total[5m]))"
          }
        ]
      },
      {
        "title": "Nodes Healthy",
        "type": "stat",
        "targets": [
          {
            "expr": "count(up{job=\"kg-nodes\"} == 1)"
          }
        ]
      },
      {
        "title": "Queries per Second",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(kg_queries_total[5m])) by (node_id)"
          }
        ]
      },
      {
        "title": "Shortcut Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "sum(rate(kg_shortcut_hits[5m])) / sum(rate(kg_queries_total[5m]))"
          }
        ]
      },
      {
        "title": "Consensus Score Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "kg_consensus_score_bucket"
          }
        ]
      }
    ]
  }
}
```

## Kubernetes Deployment

For production Kubernetes deployments, see the generated manifests below.

### KG Node Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kg-node-csv
  namespace: kg-topology
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kg-node
      domain: csv
  template:
    metadata:
      labels:
        app: kg-node
        domain: csv
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8081"
        prometheus.io/path: "/kg/metrics"
    spec:
      containers:
        - name: kg-node
          image: unifyweaver/kg-node:latest
          ports:
            - containerPort: 8081
          env:
            - name: NODE_ID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: NODE_PORT
              value: "8081"
            - name: CONSUL_HOST
              value: "consul.consul.svc.cluster.local"
            - name: INTERFACE_TOPICS
              value: "csv,delimited,tabular"
            - name: EMBEDDING_MODEL
              value: "all-MiniLM-L6-v2"
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /kg/health
              port: 8081
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /kg/health
              port: 8081
            initialDelaySeconds: 30
            periodSeconds: 30
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: embeddings
              mountPath: /app/embeddings
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: kg-csv-data
        - name: embeddings
          persistentVolumeClaim:
            claimName: kg-csv-embeddings
---
apiVersion: v1
kind: Service
metadata:
  name: kg-node-csv
  namespace: kg-topology
spec:
  selector:
    app: kg-node
    domain: csv
  ports:
    - port: 8081
      targetPort: 8081
  type: ClusterIP
```

### Consul Integration for Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: consul-config
  namespace: kg-topology
data:
  config.json: |
    {
      "datacenter": "dc1",
      "data_dir": "/consul/data",
      "log_level": "INFO",
      "server": true,
      "bootstrap_expect": 3,
      "ui_config": {
        "enabled": true
      },
      "connect": {
        "enabled": true
      }
    }
```

## Scaling Guidelines

### Horizontal Scaling

| Cluster Size | Nodes | federation_k | hierarchy_levels | Notes |
|-------------|-------|--------------|------------------|-------|
| Small | 3-5 | 3 | 1 | All nodes queried |
| Medium | 10-50 | 5-7 | 2 | Regional aggregators |
| Large | 50+ | 7-10 | 2-3 | Hierarchical federation |

### Resource Recommendations

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| KG Node (small) | 0.1-0.5 | 256Mi-512Mi | 1Gi |
| KG Node (large) | 0.5-2.0 | 512Mi-2Gi | 10Gi |
| Consul | 0.2-0.5 | 256Mi | 1Gi |
| Prometheus | 0.5-1.0 | 1Gi-4Gi | 50Gi |
| Grafana | 0.2 | 256Mi | 1Gi |

### Adaptive-K Tuning

For optimal performance, tune adaptive-k based on your workload:

```python
# For specific queries (high max_similarity)
# → fewer nodes, faster response
adaptive_k_config = {
    "base_k": 3,
    "entropy_weight": 0.3,
    "latency_weight": 0.4,  # Higher for latency-sensitive
    "consensus_weight": 0.3
}

# For exploratory queries (high variance)
# → more nodes, better coverage
adaptive_k_config = {
    "base_k": 5,
    "entropy_weight": 0.5,  # Higher for ambiguous queries
    "latency_weight": 0.2,
    "consensus_weight": 0.3
}
```

## Monitoring and Alerting

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `kg_query_latency_p99` | > 1s | Query latency too high |
| `kg_federation_success_rate` | < 95% | Federation failures |
| `kg_consensus_score_avg` | < 0.5 | Low answer confidence |
| `kg_nodes_healthy` | < 3 | Insufficient nodes |
| `kg_shortcut_hit_rate` | < 10% | Path folding not working |

### Prometheus Alerting Rules

```yaml
groups:
  - name: kg-topology
    rules:
      - alert: KGNodeDown
        expr: up{job="kg-nodes"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "KG node {{ $labels.instance }} is down"

      - alert: KGHighLatency
        expr: histogram_quantile(0.99, rate(kg_query_latency_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "KG query latency p99 > 1s"

      - alert: KGLowConsensus
        expr: avg(kg_consensus_score) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low consensus scores - answers may be unreliable"

      - alert: KGFederationFailures
        expr: rate(kg_federated_queries_failed[5m]) / rate(kg_federated_queries_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Federation failure rate > 5%"
```

## Backup and Recovery

### SQLite Database Backup

```bash
# Backup a node's database
docker exec kg-node-csv sqlite3 /app/data/kg.db ".backup /app/data/kg-backup.db"

# Copy backup to host
docker cp kg-node-csv:/app/data/kg-backup.db ./backups/

# Restore from backup
docker cp ./backups/kg-backup.db kg-node-csv:/app/data/kg-restore.db
docker exec kg-node-csv mv /app/data/kg-restore.db /app/data/kg.db
docker-compose restart kg-node-csv
```

### Embedding Files Backup

```bash
# Backup embeddings directory
docker exec kg-node-csv tar -czf /app/embeddings-backup.tar.gz -C /app embeddings/
docker cp kg-node-csv:/app/embeddings-backup.tar.gz ./backups/

# Restore embeddings
docker cp ./backups/embeddings-backup.tar.gz kg-node-csv:/app/
docker exec kg-node-csv tar -xzf /app/embeddings-backup.tar.gz -C /app
```

## Troubleshooting

### Node Not Registering with Consul

```bash
# Check Consul logs
docker logs kg-consul

# Verify node can reach Consul
docker exec kg-node-csv curl -sf http://consul:8500/v1/status/leader

# Check node registration
curl http://localhost:8500/v1/catalog/service/kg_node
```

### Federation Queries Timing Out

```bash
# Check node health
curl http://localhost:8081/kg/health

# Check federation stats
curl http://localhost:8081/kg/federation/stats

# Increase timeout
curl -X POST http://localhost:8081/kg/federated \
  -H "Content-Type: application/json" \
  -d '{"query_text": "...", "timeout_ms": 60000}'
```

### Low Consensus Scores

1. Check if nodes have different embedding models (must match)
2. Verify corpus diversity - too much overlap reduces consensus value
3. Check if data is properly indexed with `semantic_centroid` metadata

## Security Considerations

1. **TLS Required**: Always use HTTPS for remote nodes (see Phase 6 docs)
2. **Network Isolation**: Keep KG nodes in private subnet, expose only through load balancer
3. **Consul ACL**: Enable Consul ACLs in production
4. **Secrets**: Use Vault or environment variables for sensitive config

## Related Documentation

- [ROADMAP_KG_TOPOLOGY.md](../proposals/ROADMAP_KG_TOPOLOGY.md) - Phase implementation details
- [FEDERATED_QUERY_ALGEBRA.md](../proposals/FEDERATED_QUERY_ALGEBRA.md) - Aggregation strategies
- [DENSITY_SCORING_PROPOSAL.md](../proposals/DENSITY_SCORING_PROPOSAL.md) - Confidence scoring
- [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) - Federation performance optimization
- [Cross-Target Glue Phase 6](../design/cross-target-glue/05-phase6-design.md) - Deployment patterns
- [Book 7 Ch15: Production Deployment](../../education/book-07-cross-target-glue/15_deployment_production.md)
- [Book 7 Ch16: Cloud & Enterprise](../../education/book-07-cross-target-glue/16_cloud_enterprise.md)

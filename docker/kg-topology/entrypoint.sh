#!/bin/bash
set -euo pipefail

# Configuration from environment
NODE_ID="${NODE_ID:-kg-node}"
NODE_PORT="${NODE_PORT:-8080}"
CONSUL_HOST="${CONSUL_HOST:-localhost}"
CONSUL_PORT="${CONSUL_PORT:-8500}"
INTERFACE_TOPICS="${INTERFACE_TOPICS:-general}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"

echo "============================================"
echo "Starting KG Topology Node"
echo "============================================"
echo "  Node ID:    ${NODE_ID}"
echo "  Port:       ${NODE_PORT}"
echo "  Consul:     ${CONSUL_HOST}:${CONSUL_PORT}"
echo "  Topics:     ${INTERFACE_TOPICS}"
echo "  Model:      ${EMBEDDING_MODEL}"
echo "============================================"

# Wait for Consul to be available
echo "Waiting for Consul..."
max_retries=30
retry_count=0
until curl -sf "http://${CONSUL_HOST}:${CONSUL_PORT}/v1/status/leader" > /dev/null 2>&1; do
    retry_count=$((retry_count + 1))
    if [ $retry_count -ge $max_retries ]; then
        echo "ERROR: Consul not available after ${max_retries} attempts"
        exit 1
    fi
    echo "  Consul not ready, waiting... (${retry_count}/${max_retries})"
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

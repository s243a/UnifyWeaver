# KG Distributed Setup Guide

This guide covers setting up a distributed KG topology network using Kleinberg routing.

## Prerequisites

- Python 3.8+ with numpy
- UnifyWeaver core installed
- (Optional) Consul for production service discovery

## Quick Start

### 1. Single Node Setup

```python
from kg_topology_api import DistributedKGTopologyAPI

# Create a distributed KG node
api = DistributedKGTopologyAPI(
    db_path='my_kg.db',
    node_id='csv-expert-node',
    discovery_backend='local'  # Use 'consul' for production
)

# Create an interface and compute centroid
interface_id = api.create_interface(
    name='csv_expert',
    description='Expert on CSV and delimited file formats'
)

# Add Q-A pairs and compute centroid
# ... (see Phase 1-2 documentation)

# Register with discovery
api.register_node(
    host='localhost',
    port=8081,
    tags=['kg_node', 'csv_expert']
)
```

### 2. Multi-Node Simulation (Testing)

```python
from discovery_clients import LocalDiscoveryClient

# Use shared registry for multi-node simulation
LocalDiscoveryClient.clear_shared()

# Node A: CSV expert
api_a = DistributedKGTopologyAPI(
    db_path='node_a.db',
    node_id='node-a',
    discovery_backend='local',
    discovery_config={'use_shared': True}
)
api_a.register_node(host='localhost', port=8081, tags=['kg_node', 'csv'])

# Node B: JSON expert
api_b = DistributedKGTopologyAPI(
    db_path='node_b.db',
    node_id='node-b',
    discovery_backend='local',
    discovery_config={'use_shared': True}
)
api_b.register_node(host='localhost', port=8082, tags=['kg_node', 'json'])

# Query from Node A (will route to Node B if needed)
results = api_a.distributed_search(
    query_text='How do I parse JSON files?',
    max_hops=5
)
```

### 3. Production Setup with Consul

```python
api = DistributedKGTopologyAPI(
    db_path='production.db',
    node_id='prod-node-1',
    discovery_backend='consul',
    discovery_config={
        'host': 'consul.example.com',
        'port': 8500,
        'token': 'your-acl-token'  # Optional
    }
)

api.register_node(
    host='kg-node-1.example.com',
    port=8080,
    tags=['kg_node', 'production']
)
```

## Configuration Options

### Kleinberg Router Options

| Option | Default | Description |
|--------|---------|-------------|
| `alpha` | 2.0 | Link distribution exponent (higher = more local routing) |
| `max_hops` | 10 | Maximum routing hops (HTL) |
| `parallel_paths` | 1 | Number of parallel query paths |
| `similarity_threshold` | 0.5 | Minimum similarity for forwarding |
| `path_folding` | true | Create shortcuts from successful paths |

### Prolog Service Definition

```prolog
service(my_kg_node, [
    transport(http('/kg', [host('0.0.0.0'), port(8080)])),
    discovery_enabled(true),
    discovery_backend(consul),
    discovery_tags([kg_node, my_domain]),
    discovery_metadata([
        semantic_centroid("base64_encoded_vector..."),
        embedding_model('all-MiniLM-L6-v2'),
        interface_topics([topic1, topic2])
    ]),
    routing(kleinberg([
        alpha(2.0),
        max_hops(10),
        parallel_paths(1),
        similarity_threshold(0.5),
        path_folding(true)
    ]))
], [
    receive(Query),
    handle_kg_query(Query, Response),
    respond(Response)
]).
```

## HTTP Endpoints

Each KG node exposes three endpoints:

### POST /kg/query

Handle distributed queries from other nodes.

**Request:**
```json
{
    "__type": "kg_query",
    "__id": "uuid-123",
    "__routing": {
        "origin_node": "node_a",
        "htl": 8,
        "visited": ["node_a"],
        "path_folding_enabled": true
    },
    "__embedding": {
        "model": "all-MiniLM-L6-v2",
        "vector": [0.1, 0.2, ...]
    },
    "payload": {
        "query_text": "How do I parse CSV?",
        "top_k": 5
    }
}
```

**Response:**
```json
{
    "__type": "kg_response",
    "__id": "uuid-123",
    "results": [...],
    "source_node": "node_b"
}
```

### POST /kg/register

Register the node with discovery service.

**Request:**
```json
{
    "host": "localhost",
    "port": 8080,
    "tags": ["kg_node", "expert"]
}
```

### GET /kg/health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "node_id": "my-node",
    "interfaces": 3,
    "stats": {
        "total_queries": 150,
        "avg_hops": 1.5,
        "shortcuts": {"count": 25, "total_hits": 180}
    }
}
```

## Monitoring

### Query Statistics

```python
stats = api.get_query_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Average hops: {stats['avg_hops']}")
print(f"Shortcuts: {stats['shortcuts']['count']}")
```

### Router Statistics

```python
router = api.get_router()
router_stats = router.get_stats()
print(f"Cached nodes: {router_stats['cached_nodes']}")
print(f"Shortcuts: {router_stats['shortcuts']}")
```

## Maintenance

### Prune Shortcuts

Remove old or unused shortcuts:

```python
# Remove shortcuts older than 30 days with fewer than 2 hits
removed = api.prune_shortcuts(max_age_days=30, min_hits=2)
print(f"Pruned {removed} shortcuts")
```

### Deregister Node

```python
api.deregister_node()
```

## Testing

Run the integration tests:

```bash
./tests/integration/test_kg_distributed.sh
```

Run unit tests:

```bash
python -m pytest tests/core/test_kg_distributed.py -v
```

## Troubleshooting

### No nodes discovered

1. Check that nodes are registered with the same discovery backend
2. Verify tags match between registration and discovery
3. Check heartbeat/TTL expiration

### High hop counts

1. Increase `alpha` for more local routing
2. Enable path folding to create shortcuts
3. Add more nodes with diverse interface centroids

### Slow queries

1. Enable parallel paths: `parallel_paths=3`
2. Lower similarity threshold for faster forwarding
3. Check network latency between nodes

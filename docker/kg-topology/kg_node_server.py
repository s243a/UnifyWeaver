#!/usr/bin/env python3
"""
KG Topology Node Server

A Flask server that provides semantic search with Kleinberg routing
and federated query support.

Usage:
    python kg_node_server.py --node-id csv-node --port 8081 --topics csv,delimited
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import Optional

from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import KG topology modules
try:
    from kg_topology_api import DistributedKGTopologyAPI
    from kleinberg_router import KleinbergRouter, KGNode
    from discovery_clients import create_discovery_client, ServiceInstance
    from federated_query import (
        FederatedQueryEngine,
        AggregationConfig,
        AggregationStrategy
    )
except ImportError as e:
    logger.error(f"Failed to import KG topology modules: {e}")
    logger.error("Make sure all runtime modules are in the Python path")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)

# Global state (initialized in main)
api: Optional[DistributedKGTopologyAPI] = None
router: Optional[KleinbergRouter] = None
discovery_client = None
node_config = {}
start_time = time.time()

# Metrics counters
metrics = {
    'queries_total': 0,
    'queries_success': 0,
    'queries_failed': 0,
    'federated_queries_total': 0,
    'federated_queries_success': 0,
    'shortcut_hits': 0,
    'total_latency_ms': 0.0
}


@app.route('/kg/health', methods=['GET'])
def health():
    """Health check endpoint for Consul and load balancers."""
    uptime = time.time() - start_time
    return jsonify({
        'status': 'healthy',
        'node_id': node_config.get('node_id', 'unknown'),
        'uptime_seconds': round(uptime, 2),
        'topics': node_config.get('topics', []),
        'embedding_model': node_config.get('embedding_model', 'unknown'),
        'routing_stats': router.get_stats() if router else {}
    })


@app.route('/kg/metrics', methods=['GET'])
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    lines = []
    node_id = node_config.get('node_id', 'unknown')

    lines.append(f'# HELP kg_queries_total Total queries received')
    lines.append(f'# TYPE kg_queries_total counter')
    lines.append(f'kg_queries_total{{node_id="{node_id}"}} {metrics["queries_total"]}')

    lines.append(f'# HELP kg_queries_success Successful queries')
    lines.append(f'# TYPE kg_queries_success counter')
    lines.append(f'kg_queries_success{{node_id="{node_id}"}} {metrics["queries_success"]}')

    lines.append(f'# HELP kg_federated_queries_total Total federated queries')
    lines.append(f'# TYPE kg_federated_queries_total counter')
    lines.append(f'kg_federated_queries_total{{node_id="{node_id}"}} {metrics["federated_queries_total"]}')

    lines.append(f'# HELP kg_shortcut_hits Path folding shortcut hits')
    lines.append(f'# TYPE kg_shortcut_hits counter')
    lines.append(f'kg_shortcut_hits{{node_id="{node_id}"}} {metrics["shortcut_hits"]}')

    avg_latency = (
        metrics['total_latency_ms'] / max(1, metrics['queries_total'])
    )
    lines.append(f'# HELP kg_query_latency_avg_ms Average query latency in milliseconds')
    lines.append(f'# TYPE kg_query_latency_avg_ms gauge')
    lines.append(f'kg_query_latency_avg_ms{{node_id="{node_id}"}} {avg_latency:.2f}')

    return '\n'.join(lines), 200, {'Content-Type': 'text/plain'}


@app.route('/kg/query', methods=['POST'])
def handle_query():
    """Handle a semantic search query."""
    start = time.time()
    metrics['queries_total'] += 1

    try:
        data = request.get_json()
        query_text = data.get('query_text', '')
        top_k = data.get('top_k', 5)

        if not query_text:
            return jsonify({'error': 'query_text is required'}), 400

        # Check for shortcut (path folding)
        if router:
            shortcut = router.check_shortcut(query_text)
            if shortcut:
                metrics['shortcut_hits'] += 1
                return jsonify({
                    'redirect': shortcut,
                    'shortcut_hit': True,
                    'node_id': node_config['node_id']
                })

        # Perform local search
        if api:
            results = api.semantic_search(
                query_text=query_text,
                model_name=node_config.get('embedding_model', 'all-MiniLM-L6-v2'),
                top_k=top_k
            )

            elapsed_ms = (time.time() - start) * 1000
            metrics['queries_success'] += 1
            metrics['total_latency_ms'] += elapsed_ms

            return jsonify({
                'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in results],
                'node_id': node_config['node_id'],
                'query_time_ms': round(elapsed_ms, 2)
            })
        else:
            return jsonify({
                'error': 'API not initialized',
                'node_id': node_config.get('node_id', 'unknown')
            }), 503

    except Exception as e:
        metrics['queries_failed'] += 1
        logger.exception(f"Query failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/kg/federated', methods=['POST'])
def handle_federated_query():
    """Handle a federated query across multiple nodes."""
    start = time.time()
    metrics['federated_queries_total'] += 1

    try:
        data = request.get_json()
        query_text = data.get('query_text', '')
        top_k = data.get('top_k', 5)
        federation_k = data.get('federation_k', 3)
        timeout_ms = data.get('timeout_ms', 30000)
        strategy = data.get('strategy', 'sum')

        if not query_text:
            return jsonify({'error': 'query_text is required'}), 400

        if not router:
            return jsonify({'error': 'Router not initialized'}), 503

        # Create federated query engine
        config = AggregationConfig(
            strategy=AggregationStrategy[strategy.upper()],
            dedup_key='answer_id'
        )
        engine = FederatedQueryEngine(router, aggregation_config=config)

        # Generate embedding for query
        if api:
            embedding = api.get_embedding(
                query_text,
                node_config.get('embedding_model', 'all-MiniLM-L6-v2')
            )
        else:
            # Fallback: cannot generate embedding without API
            return jsonify({'error': 'API not initialized for embedding'}), 503

        # Execute federated query
        response = engine.federated_query(
            query_text=query_text,
            embedding=embedding,
            federation_k=federation_k,
            timeout_ms=timeout_ms
        )

        elapsed_ms = (time.time() - start) * 1000
        metrics['federated_queries_success'] += 1

        return jsonify({
            'results': [r.to_dict() for r in response.results],
            'nodes_queried': response.nodes_queried,
            'nodes_responded': response.nodes_responded,
            'query_time_ms': round(elapsed_ms, 2),
            'aggregation_strategy': strategy
        })

    except Exception as e:
        logger.exception(f"Federated query failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/kg/federation/stats', methods=['GET'])
def federation_stats():
    """Get federation statistics."""
    if router:
        return jsonify({
            'node_id': node_config.get('node_id', 'unknown'),
            'routing_stats': router.get_stats(),
            'metrics': metrics
        })
    else:
        return jsonify({'error': 'Router not initialized'}), 503


@app.route('/kg/register', methods=['POST'])
def register_with_discovery():
    """Manually trigger re-registration with discovery service."""
    try:
        if discovery_client and api:
            register_node()
            return jsonify({'status': 'registered', 'node_id': node_config['node_id']})
        else:
            return jsonify({'error': 'Discovery client not initialized'}), 503
    except Exception as e:
        logger.exception(f"Registration failed: {e}")
        return jsonify({'error': str(e)}), 500


def register_node():
    """Register this node with the discovery service."""
    if not discovery_client:
        logger.warning("No discovery client configured")
        return

    # Compute semantic centroid from local data
    centroid = None
    if api:
        try:
            centroid = api.compute_interface_centroid()
            if centroid is not None:
                # Encode as base64 for transmission
                import numpy as np
                centroid_bytes = centroid.astype(np.float32).tobytes()
                centroid_b64 = base64.b64encode(centroid_bytes).decode('utf-8')
            else:
                centroid_b64 = ""
        except Exception as e:
            logger.warning(f"Failed to compute centroid: {e}")
            centroid_b64 = ""
    else:
        centroid_b64 = ""

    # Create service instance
    instance = ServiceInstance(
        service_id=node_config['node_id'],
        service_name='kg_node',
        host=node_config.get('host', '0.0.0.0'),
        port=node_config['port'],
        tags=['kg_node'] + node_config.get('topics', []),
        metadata={
            'semantic_centroid': centroid_b64,
            'embedding_model': node_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            'interface_topics': json.dumps(node_config.get('topics', []))
        }
    )

    discovery_client.register(instance)
    logger.info(f"Registered with discovery service: {node_config['node_id']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='KG Topology Node Server')
    parser.add_argument('--node-id', required=True, help='Unique node identifier')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--consul-host', default='localhost', help='Consul host')
    parser.add_argument('--consul-port', type=int, default=8500, help='Consul port')
    parser.add_argument('--topics', default='general', help='Comma-separated topics')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Embedding model')
    parser.add_argument('--db-path', default='/app/data/kg.db', help='SQLite database path')
    parser.add_argument('--embeddings-path', default='/app/embeddings', help='Embeddings directory')
    return parser.parse_args()


def main():
    """Initialize and start the KG node server."""
    global api, router, discovery_client, node_config

    args = parse_args()

    # Parse topics
    topics = [t.strip() for t in args.topics.split(',') if t.strip()]

    # Store configuration
    node_config = {
        'node_id': args.node_id,
        'port': args.port,
        'host': args.host,
        'topics': topics,
        'embedding_model': args.embedding_model
    }

    logger.info(f"Initializing KG node: {args.node_id}")

    # Create discovery client
    try:
        discovery_client = create_discovery_client(
            'consul',
            host=args.consul_host,
            port=args.consul_port
        )
        logger.info(f"Connected to Consul at {args.consul_host}:{args.consul_port}")
    except Exception as e:
        logger.warning(f"Failed to create discovery client: {e}")
        logger.warning("Running without service discovery")
        discovery_client = None

    # Create Kleinberg router
    try:
        router = KleinbergRouter(
            discovery_client=discovery_client,
            alpha=2.0,
            max_hops=10,
            similarity_threshold=0.5,
            path_folding_enabled=True
        )
        logger.info("Kleinberg router initialized")
    except Exception as e:
        logger.warning(f"Failed to create router: {e}")
        router = None

    # Initialize KG topology API
    try:
        api = DistributedKGTopologyAPI(
            db_path=args.db_path,
            embeddings_path=args.embeddings_path,
            discovery_client=discovery_client,
            enable_distributed=True
        )
        logger.info(f"KG API initialized with database: {args.db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize KG API: {e}")
        api = None

    # Register with discovery service
    if discovery_client:
        try:
            register_node()
        except Exception as e:
            logger.warning(f"Failed to register with discovery: {e}")

    # Start Flask server
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()

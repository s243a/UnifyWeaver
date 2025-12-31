# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Discovery client implementations for distributed KG topology.

"""
Discovery client implementations for distributed KG topology (Phase 3).

Provides abstract base class and implementations for service discovery:
- LocalDiscoveryClient: In-memory for testing and single-machine deployments
- ConsulDiscoveryClient: HashiCorp Consul integration
- EtcdDiscoveryClient: CoreOS etcd integration (stub)
- DNSDiscoveryClient: DNS-based discovery (stub)

See: docs/proposals/ROADMAP_KG_TOPOLOGY.md (Phase 3)
     docs/CLIENT_SERVER_DESIGN.md (Phase 7: Service Discovery)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto
import json
import threading
import time
import urllib.request
import urllib.error


class HealthStatus(Enum):
    """Health status of a service instance."""
    HEALTHY = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class ServiceInstance:
    """Represents a discovered service instance."""
    service_id: str
    service_name: str
    host: str
    port: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'service_id': self.service_id,
            'service_name': self.service_name,
            'host': self.host,
            'port': self.port,
            'tags': self.tags,
            'metadata': self.metadata,
            'health_status': self.health_status.name,
            'last_heartbeat': self.last_heartbeat
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInstance':
        """Create from dictionary."""
        health_str = data.get('health_status', 'UNKNOWN')
        health = HealthStatus[health_str] if isinstance(health_str, str) else HealthStatus.UNKNOWN
        return cls(
            service_id=data['service_id'],
            service_name=data['service_name'],
            host=data['host'],
            port=data['port'],
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            health_status=health,
            last_heartbeat=data.get('last_heartbeat', time.time())
        )


class DiscoveryClient(ABC):
    """
    Abstract base class for service discovery clients.

    Implementations must provide:
    - register: Register a service instance
    - deregister: Remove a service instance
    - discover: Find service instances by name and tags
    - heartbeat: Send heartbeat to maintain registration
    """

    @abstractmethod
    def register(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: int = 60
    ) -> bool:
        """
        Register a service instance.

        Args:
            service_name: Name of the service (e.g., 'kg_topology')
            service_id: Unique ID for this instance
            host: Host address
            port: Port number
            tags: List of tags for filtering
            metadata: Additional metadata (including semantic_centroid)
            ttl: Time-to-live in seconds for heartbeat

        Returns:
            True if registration succeeded
        """
        pass

    @abstractmethod
    def deregister(self, service_id: str) -> bool:
        """
        Deregister a service instance.

        Args:
            service_id: ID of instance to deregister

        Returns:
            True if deregistration succeeded
        """
        pass

    @abstractmethod
    def discover(
        self,
        service_name: str,
        tags: List[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """
        Discover service instances.

        Args:
            service_name: Name of service to find
            tags: Filter by tags (all tags must match)
            healthy_only: Only return healthy instances

        Returns:
            List of matching service instances
        """
        pass

    @abstractmethod
    def heartbeat(self, service_id: str) -> bool:
        """
        Send heartbeat for a registered service.

        Args:
            service_id: ID of instance to heartbeat

        Returns:
            True if heartbeat succeeded
        """
        pass


class LocalDiscoveryClient(DiscoveryClient):
    """
    In-memory discovery client for testing and single-machine deployments.

    Thread-safe implementation using a lock for concurrent access.
    Supports TTL-based expiration of instances.
    """

    # Class-level shared registry (singleton pattern for testing)
    _shared_instances: Dict[str, ServiceInstance] = {}
    _shared_lock = threading.Lock()
    _use_shared: bool = False

    def __init__(self, use_shared: bool = False):
        """
        Initialize local discovery client.

        Args:
            use_shared: If True, use shared class-level registry
                       (useful for multi-node simulation in tests)
        """
        self._use_shared = use_shared
        if not use_shared:
            self._instances: Dict[str, ServiceInstance] = {}
            self._lock = threading.Lock()
            self._ttls: Dict[str, int] = {}
        else:
            self._instances = LocalDiscoveryClient._shared_instances
            self._lock = LocalDiscoveryClient._shared_lock
            self._ttls: Dict[str, int] = {}

    def register(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: int = 60
    ) -> bool:
        """Register a service instance in local registry."""
        instance = ServiceInstance(
            service_id=service_id,
            service_name=service_name,
            host=host,
            port=port,
            tags=tags or [],
            metadata=metadata or {},
            health_status=HealthStatus.HEALTHY,
            last_heartbeat=time.time()
        )

        with self._lock:
            self._instances[service_id] = instance
            self._ttls[service_id] = ttl

        return True

    def deregister(self, service_id: str) -> bool:
        """Remove a service instance from local registry."""
        with self._lock:
            if service_id in self._instances:
                del self._instances[service_id]
                self._ttls.pop(service_id, None)
                return True
        return False

    def discover(
        self,
        service_name: str,
        tags: List[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Find service instances in local registry."""
        results = []
        current_time = time.time()

        with self._lock:
            for instance in self._instances.values():
                # Filter by service name
                if instance.service_name != service_name:
                    continue

                # Filter by health (check TTL expiration)
                if healthy_only:
                    ttl = self._ttls.get(instance.service_id, 60)
                    if current_time - instance.last_heartbeat > ttl:
                        instance.health_status = HealthStatus.UNHEALTHY
                        continue

                # Filter by tags (all specified tags must be present)
                if tags:
                    if not all(t in instance.tags for t in tags):
                        continue

                results.append(instance)

        return results

    def heartbeat(self, service_id: str) -> bool:
        """Update heartbeat timestamp for a service."""
        with self._lock:
            if service_id in self._instances:
                self._instances[service_id].last_heartbeat = time.time()
                self._instances[service_id].health_status = HealthStatus.HEALTHY
                return True
        return False

    def clear(self):
        """Clear all registered instances (for testing)."""
        with self._lock:
            self._instances.clear()
            self._ttls.clear()

    @classmethod
    def clear_shared(cls):
        """Clear the shared registry (for testing)."""
        with cls._shared_lock:
            cls._shared_instances.clear()


class ConsulDiscoveryClient(DiscoveryClient):
    """
    HashiCorp Consul-based service discovery client.

    Uses Consul's HTTP API for service registration and discovery.
    Supports health checks and TTL-based heartbeats.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8500,
        token: str = None,
        datacenter: str = None
    ):
        """
        Initialize Consul discovery client.

        Args:
            host: Consul agent host
            port: Consul agent port
            token: ACL token for authentication
            datacenter: Datacenter to use
        """
        self.host = host
        self.port = port
        self.token = token
        self.datacenter = datacenter
        self.base_url = f'http://{host}:{port}/v1'

    def _make_request(
        self,
        method: str,
        path: str,
        data: Any = None,
        params: Dict[str, str] = None
    ) -> Optional[Any]:
        """Make HTTP request to Consul API."""
        url = f'{self.base_url}{path}'

        if params:
            query = '&'.join(f'{k}={v}' for k, v in params.items())
            url = f'{url}?{query}'

        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['X-Consul-Token'] = self.token

        body = json.dumps(data).encode() if data else None

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method
            )
            response = urllib.request.urlopen(req, timeout=10)

            if response.status == 200:
                content = response.read()
                if content:
                    return json.loads(content)
                return True
            return None

        except urllib.error.URLError as e:
            # Log error in production
            return None
        except json.JSONDecodeError:
            return None

    def register(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: int = 60
    ) -> bool:
        """Register service with Consul agent."""
        payload = {
            'ID': service_id,
            'Name': service_name,
            'Address': host,
            'Port': port,
            'Tags': tags or [],
            'Meta': metadata or {},
            'Check': {
                'TTL': f'{ttl}s',
                'DeregisterCriticalServiceAfter': f'{ttl * 3}s'
            }
        }

        result = self._make_request('PUT', '/agent/service/register', data=payload)

        if result is not None:
            # Initial pass to mark healthy
            self.heartbeat(service_id)
            return True
        return False

    def deregister(self, service_id: str) -> bool:
        """Deregister service from Consul agent."""
        result = self._make_request('PUT', f'/agent/service/deregister/{service_id}')
        return result is not None

    def discover(
        self,
        service_name: str,
        tags: List[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover services from Consul catalog."""
        params = {}
        if healthy_only:
            params['passing'] = 'true'
        if self.datacenter:
            params['dc'] = self.datacenter

        # Build URL with optional tag filtering
        path = f'/health/service/{service_name}'
        if tags:
            for tag in tags:
                params[f'tag'] = tag  # Note: Consul uses repeated tag params

        data = self._make_request('GET', path, params=params)

        if not data:
            return []

        results = []
        for entry in data:
            service = entry.get('Service', {})
            checks = entry.get('Checks', [])

            # Determine health from checks
            health = HealthStatus.HEALTHY
            for check in checks:
                if check.get('Status') != 'passing':
                    health = HealthStatus.UNHEALTHY
                    break

            instance = ServiceInstance(
                service_id=service.get('ID', ''),
                service_name=service.get('Service', service_name),
                host=service.get('Address', ''),
                port=service.get('Port', 0),
                tags=service.get('Tags', []),
                metadata=service.get('Meta', {}),
                health_status=health
            )
            results.append(instance)

        return results

    def heartbeat(self, service_id: str) -> bool:
        """Send TTL check pass to Consul."""
        result = self._make_request('PUT', f'/agent/check/pass/service:{service_id}')
        return result is not None


class EtcdDiscoveryClient(DiscoveryClient):
    """
    CoreOS etcd-based service discovery client (stub).

    TODO: Implement etcd v3 API integration.
    """

    def __init__(self, hosts: List[str] = None):
        """Initialize etcd client."""
        self.hosts = hosts or ['localhost:2379']
        # TODO: Initialize etcd client

    def register(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: int = 60
    ) -> bool:
        """Register service with etcd."""
        raise NotImplementedError("etcd discovery not yet implemented")

    def deregister(self, service_id: str) -> bool:
        """Deregister service from etcd."""
        raise NotImplementedError("etcd discovery not yet implemented")

    def discover(
        self,
        service_name: str,
        tags: List[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover services from etcd."""
        raise NotImplementedError("etcd discovery not yet implemented")

    def heartbeat(self, service_id: str) -> bool:
        """Send heartbeat to etcd."""
        raise NotImplementedError("etcd discovery not yet implemented")


class DNSDiscoveryClient(DiscoveryClient):
    """
    DNS-based service discovery client (stub).

    TODO: Implement DNS SRV record queries.
    """

    def __init__(self, domain: str = 'service.consul'):
        """Initialize DNS client."""
        self.domain = domain
        # TODO: Initialize DNS resolver

    def register(
        self,
        service_name: str,
        service_id: str,
        host: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        ttl: int = 60
    ) -> bool:
        """DNS-based discovery typically doesn't support dynamic registration."""
        raise NotImplementedError("DNS discovery is read-only")

    def deregister(self, service_id: str) -> bool:
        """DNS-based discovery typically doesn't support deregistration."""
        raise NotImplementedError("DNS discovery is read-only")

    def discover(
        self,
        service_name: str,
        tags: List[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover services via DNS SRV records."""
        raise NotImplementedError("DNS discovery not yet implemented")

    def heartbeat(self, service_id: str) -> bool:
        """DNS-based discovery doesn't use heartbeats."""
        return True  # No-op


def create_discovery_client(backend: str, **kwargs) -> DiscoveryClient:
    """
    Factory function to create discovery client by backend name.

    Args:
        backend: Backend type ('local', 'consul', 'etcd', 'dns')
        **kwargs: Backend-specific configuration

    Returns:
        Configured discovery client

    Raises:
        ValueError: If backend is unknown
    """
    backends = {
        'local': LocalDiscoveryClient,
        'consul': ConsulDiscoveryClient,
        'etcd': EtcdDiscoveryClient,
        'dns': DNSDiscoveryClient,
    }

    if backend not in backends:
        raise ValueError(f"Unknown discovery backend: {backend}. "
                        f"Available: {list(backends.keys())}")

    return backends[backend](**kwargs)

# Playbook: Service Deployment and Lifecycle Management

## Audience
This playbook demonstrates UnifyWeaver's deployment_glue module for deploying and managing services via SSH, Docker, Kubernetes, and cloud functions.

## Overview
The `deployment_glue` module provides:
- SSH-based deployment with agent forwarding
- Service lifecycle management (start/stop/restart)
- Docker/Kubernetes manifest generation
- Cloud function deployment (AWS Lambda, GCP, Azure)
- Secrets management integration (Vault, AWS, Azure, GCP)

## When to Use

✅ **Use deployment_glue when:**
- Deploying services to remote hosts
- Managing service lifecycles
- Generating Docker/Kubernetes configs
- Need automated deployment pipelines
- Integrating with cloud platforms

## Agent Inputs

1. **Glue Module** – `src/unifyweaver/glue/deployment_glue.pl`

## Execution Guidance

### Example 1: Declare Service

```prolog
% Load deployment glue
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare a service
:- declare_service(myapp, [
    port(8080),
    health_check('/health'),
    restart_policy(always)
]).

% Declare deployment method
:- declare_deploy_method(myapp, ssh, [
    host('prod.example.com'),
    user(deploy),
    path('/opt/myapp')
]).
```

### Example 2: Generate Deployment Script

```prolog
% Generate deployment script
?- deployment_glue:generate_deploy_script(myapp, ssh, Script).
```

**Generated Output:**
```bash
#!/bin/bash
# Deploy myapp via SSH
ssh -A deploy@prod.example.com << 'EOF'
cd /opt/myapp
./deploy.sh
systemctl restart myapp
EOF
```

### Example 3: Docker Deployment

```prolog
% Declare Docker deployment
:- declare_deploy_method(myapp, docker, [
    image('myapp:latest'),
    ports(['8080:8080']),
    env(['DB_HOST=db.example.com'])
]).

% Generate Dockerfile
?- deployment_glue:generate_dockerfile(myapp, Dockerfile).
```

### Example 4: Kubernetes Deployment

```prolog
% Declare Kubernetes deployment
:- declare_deploy_method(myapp, kubernetes, [
    namespace(production),
    replicas(3),
    image('myapp:v1.0')
]).

% Generate K8s manifests
?- deployment_glue:generate_k8s_deployment(myapp, Manifest).
```

## Key Features

**SSH Deployment:**
- Agent forwarding support
- Change detection
- Automatic redeployment
- Security validation

**Container Deployment (Experimental):**
- Docker/Docker Compose generation
- Kubernetes manifests (Deployment, Service, Ingress)
- Helm chart generation
- Registry authentication

**Secrets Management (Experimental):**
- HashiCorp Vault integration
- AWS Secrets Manager
- Azure Key Vault
- GCP Secret Manager

**Cloud Functions (Experimental):**
- AWS Lambda deployment
- Google Cloud Functions
- Azure Functions
- API Gateway integration

## See Also

- `playbooks/network_glue_playbook.md` - Network communication
- `playbooks/cross_target_glue_playbook.md` - Cross-language pipelines

## Summary

**Key Concepts:**
- ✅ Service deployment automation
- ✅ SSH, Docker, Kubernetes support
- ✅ Cloud function deployment
- ✅ Secrets management integration
- ✅ Lifecycle management (start/stop/restart)

**Note**: Container, secrets, and cloud features are experimental - code generation tested, but not deployment functionality.

# Playbook: Service Deployment and Lifecycle Management

## Audience
This playbook is a high-level guide for coding agents. It demonstrates UnifyWeaver's deployment_glue module for deploying and managing services via SSH, Docker, Kubernetes, and cloud functions.

## Workflow Overview
Use deployment_glue for service deployment:
1. Declare services with configuration (ports, health checks, restart policies)
2. Declare deployment methods (SSH, Docker, Kubernetes, cloud functions)
3. Generate deployment scripts and manifests
4. Execute deployment workflows

## Agent Inputs
Reference the following artifacts:
1. **Glue Module** – `src/unifyweaver/glue/deployment_glue.pl` contains all deployment predicates
2. **Module Documentation** – See module header comments for API details

## Key Features

### SSH Deployment
- Agent forwarding support
- Change detection and automatic redeployment
- Security validation

### Container Deployment (Experimental)
- Docker/Docker Compose generation
- Kubernetes manifests
- Helm charts

### Secrets Management (Experimental)
- HashiCorp Vault, AWS Secrets Manager
- Azure Key Vault, GCP Secret Manager

### Cloud Functions (Experimental)
- AWS Lambda, Google Cloud Functions, Azure Functions

## Execution Guidance

Consult the module directly for predicate usage:

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare a service
:- declare_service(myapp, [port(8080), health_check('/health')]).

% Declare SSH deployment
:- declare_deploy_method(myapp, ssh, [
    host('prod.example.com'),
    user(deploy),
    path('/opt/myapp')
]).

% Generate deployment script
?- generate_deploy_script(myapp, ssh, Script).
```

## Expected Outcome
- Services declared successfully
- Deployment scripts generated
- Manifests created for container platforms
- Ready for deployment execution

## Citations
[1] src/unifyweaver/glue/deployment_glue.pl

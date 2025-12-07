# Test Coverage Documentation

This document describes the test coverage for UnifyWeaver's cross-target glue system, with particular attention to what is and is NOT tested.

## Overview

| Phase | Module | Unit Tests | Integration Tests | Production Ready |
|-------|--------|------------|-------------------|------------------|
| 1-5 | Core glue modules | ✅ Yes | ⚠️ Partial | ✅ Yes |
| 6a-d | deployment_glue.pl | ✅ Yes | ⚠️ Partial | ✅ Yes |
| 7a | Container deployment | ✅ Yes | ❌ No | ⚠️ Experimental |
| 7b | Secrets management | ✅ Yes | ❌ No | ⚠️ Experimental |

## Test Categories

### Unit Tests (What We Have)

Unit tests verify that predicates:
- Accept correct input and reject invalid input
- Store and retrieve configuration correctly
- Generate expected output strings/structures
- Handle edge cases in logic

**These tests run without external dependencies.**

### Integration Tests (What's Needed for Production)

Integration tests would verify:
- Generated code actually executes
- External services respond correctly
- End-to-end workflows complete successfully

**These require actual infrastructure (Docker, Kubernetes, SSH servers, etc.)**

---

## Phase 6: Deployment Glue

### Phase 6a-b: Deployment Foundation

| Feature | Unit Test | Integration Test | Notes |
|---------|-----------|------------------|-------|
| Service declarations | ✅ | N/A | Configuration storage |
| Deploy method config | ✅ | N/A | Configuration storage |
| Source tracking | ✅ | N/A | Configuration storage |
| Security validation | ✅ | N/A | Logic validation |
| SSH script generation | ✅ | ❌ | Generates scripts, doesn't execute |
| Systemd unit generation | ✅ | ❌ | Generates units, doesn't install |
| Multi-host deployment | ✅ | ❌ | Generates scripts for each host |
| Rollback scripts | ✅ | ❌ | Generates scripts, doesn't execute |

**To fully test:** Would need SSH access to test servers, systemd environments.

### Phase 6c: Error Handling

| Feature | Unit Test | Integration Test | Notes |
|---------|-----------|------------------|-------|
| Retry policy config | ✅ | N/A | Configuration storage |
| Retry execution | ✅ | ⚠️ | Tests with mock predicates |
| Fallback config | ✅ | N/A | Configuration storage |
| Fallback execution | ✅ | ⚠️ | Tests with mock predicates |
| Circuit breaker config | ✅ | N/A | Configuration storage |
| Circuit state transitions | ✅ | N/A | State machine logic |
| Timeout config | ✅ | N/A | Configuration storage |
| Protected call | ✅ | ⚠️ | Tests with mock predicates |

**Status:** Reasonably well tested. Mock predicates simulate success/failure scenarios.

### Phase 6d: Monitoring

| Feature | Unit Test | Integration Test | Notes |
|---------|-----------|------------------|-------|
| Health check config | ✅ | N/A | Configuration storage |
| Health status tracking | ✅ | ❌ | Would need HTTP endpoints |
| Metrics config | ✅ | N/A | Configuration storage |
| Metric recording | ✅ | N/A | In-memory storage |
| Prometheus export | ✅ | ❌ | Format verified, not scraped |
| Logging config | ✅ | N/A | Configuration storage |
| Log events | ✅ | N/A | In-memory storage |
| Alert config | ✅ | N/A | Configuration storage |
| Alert triggering | ✅ | ❌ | Doesn't send real notifications |

**To fully test:** Would need Prometheus server, actual HTTP services, notification endpoints.

---

## Phase 7a: Container Deployment [EXPERIMENTAL]

**⚠️ WARNING: All Phase 7a features are experimental.**

### Test Coverage

| Feature | What's Tested | What's NOT Tested |
|---------|---------------|-------------------|
| `declare_docker_config/2` | Config stored/retrieved | N/A |
| `generate_dockerfile/3` | Output contains expected strings | Dockerfile builds successfully |
| `generate_dockerignore/3` | Output contains expected patterns | N/A |
| `docker_image_tag/2` | Tag format correct | Image exists |
| `build_docker_image/3` | Command string generated | Docker actually builds |
| `push_docker_image/3` | Command string generated | Push succeeds |
| `declare_compose_config/2` | Config stored/retrieved | N/A |
| `generate_docker_compose/3` | YAML structure correct | `docker-compose up` works |
| `declare_k8s_config/2` | Config stored/retrieved | N/A |
| `generate_k8s_deployment/3` | YAML contains required fields | `kubectl apply` succeeds |
| `generate_k8s_service/3` | YAML contains required fields | Service routes traffic |
| `generate_k8s_configmap/3` | YAML contains required fields | ConfigMap created |
| `generate_k8s_ingress/3` | YAML contains required fields | Ingress routes traffic |
| `generate_helm_chart/3` | Chart structure created | `helm install` works |
| `declare_registry/2` | Config stored/retrieved | N/A |
| `login_registry/2` | Command string generated | Login succeeds |
| `deploy_to_k8s/3` | Commands generated | Deployment succeeds |
| `scale_k8s_deployment/4` | Command generated | Scaling works |

### What Unit Tests Verify

```prolog
% Example: test_generate_dockerfile_python
test_generate_dockerfile_python :-
    declare_service(test_svc, [target(python), port(8080)]),
    declare_docker_config(test_svc, [base_image('python:3.11-slim')]),
    generate_dockerfile(test_svc, [], Dockerfile),
    % These checks verify string content, NOT that it builds:
    sub_atom(Dockerfile, _, _, _, 'FROM python:3.11-slim'),
    sub_atom(Dockerfile, _, _, _, 'pip install'),
    sub_atom(Dockerfile, _, _, _, 'EXPOSE 8080').
```

### What Would Be Needed for Full Testing

1. **Docker Integration Tests**
   ```bash
   # Verify Dockerfile builds
   generate_dockerfile(svc, [], DF),
   write_to_file('Dockerfile', DF),
   docker build -t test:latest .

   # Verify image runs
   docker run --rm test:latest --version
   ```

2. **Kubernetes Integration Tests**
   ```bash
   # Verify manifests are valid
   generate_k8s_deployment(svc, [], Manifest),
   kubectl apply --dry-run=client -f - <<< "$Manifest"

   # Full deployment test (requires cluster)
   kubectl apply -f - <<< "$Manifest"
   kubectl rollout status deployment/svc
   ```

3. **Registry Integration Tests**
   ```bash
   # Verify login works
   login_registry(reg, login_command(_, Cmd)),
   eval "$Cmd"  # Actually execute login

   # Verify push works
   docker push $IMAGE_TAG
   ```

### Recommended Validation Before Production

Before using Phase 7a in production:

1. **Manual Dockerfile validation:**
   ```bash
   # Generate and build
   swipl -g "generate_dockerfile(my_svc, [], DF), write(DF)" -t halt > Dockerfile
   docker build -t test .
   ```

2. **Kubernetes dry-run:**
   ```bash
   # Generate and validate
   swipl -g "generate_k8s_deployment(my_svc, [], M), write(M)" -t halt | \
     kubectl apply --dry-run=client -f -
   ```

3. **Compose validation:**
   ```bash
   # Generate and check
   swipl -g "generate_docker_compose(proj, [], C), write(C)" -t halt > docker-compose.yml
   docker-compose config  # Validates syntax
   ```

---

## Phase 7b: Secrets Management [EXPERIMENTAL]

**⚠️ WARNING: All Phase 7b features are experimental.**

### Test Coverage

| Feature | What's Tested | What's NOT Tested |
|---------|---------------|-------------------|
| `declare_secret_source/2` | Config stored/retrieved | N/A |
| `declare_vault_config/2` | Config stored/retrieved | Vault server connectivity |
| `generate_vault_read/4` | CLI command string generated | Vault authentication works |
| `generate_vault_agent_config/3` | Config file structure | Agent actually runs |
| `declare_aws_secrets_config/2` | Config stored/retrieved | AWS credentials valid |
| `generate_aws_secret_read/4` | CLI command string generated | Secret retrieved from AWS |
| `declare_azure_keyvault_config/2` | Config stored/retrieved | Azure credentials valid |
| `generate_azure_secret_read/4` | CLI command string generated | Secret retrieved from Azure |
| `declare_gcp_secrets_config/2` | Config stored/retrieved | GCP credentials valid |
| `generate_gcp_secret_read/4` | CLI command string generated | Secret retrieved from GCP |
| `declare_service_secrets/2` | Config stored/retrieved | N/A |
| `generate_secret_env_script/3` | Script structure correct | Script executes successfully |
| `generate_k8s_secret/3` | YAML structure correct | `kubectl apply` succeeds |
| `generate_k8s_external_secret/3` | YAML structure correct | ExternalSecrets operator works |
| `resolve_secret/4` | Command for each source type | Secret values retrieved |
| `list_secrets/3` | List commands generated | Secrets listed from sources |

### What Unit Tests Verify

```prolog
% Example: test_generate_vault_read_token
test_generate_vault_read_token :-
    declare_vault_config(test_vault, [url('https://vault.example.com'), auth_method(token)]),
    generate_vault_read(test_vault, 'secret/data/myapp', [], Command),
    % Verifies command structure, NOT that Vault is accessible:
    sub_atom(Command, _, _, _, 'vault kv get'),
    sub_atom(Command, _, _, _, '-address=https://vault.example.com').
```

### What Would Be Needed for Full Testing

1. **HashiCorp Vault Integration Tests**
   ```bash
   # Start test Vault server
   vault server -dev

   # Write test secret
   vault kv put secret/test password=secret123

   # Verify generated command works
   eval "$(generate_vault_read(cfg, 'secret/test', [], Cmd), write(Cmd))"
   ```

2. **AWS Secrets Manager Integration Tests**
   ```bash
   # Requires valid AWS credentials
   aws secretsmanager create-secret --name test/secret --secret-string '{"key":"value"}'

   # Verify generated command works
   eval "$(generate_aws_secret_read(cfg, 'test/secret', [], Cmd), write(Cmd))"
   ```

3. **Azure Key Vault Integration Tests**
   ```bash
   # Requires Azure subscription and Key Vault
   az keyvault secret set --vault-name myvault --name testsecret --value secret123

   # Verify generated command works
   eval "$(generate_azure_secret_read(cfg, testsecret, [], Cmd), write(Cmd))"
   ```

4. **GCP Secret Manager Integration Tests**
   ```bash
   # Requires GCP project
   echo -n "secret123" | gcloud secrets create test-secret --data-file=-

   # Verify generated command works
   eval "$(generate_gcp_secret_read(cfg, 'test-secret', [], Cmd), write(Cmd))"
   ```

### Recommended Validation Before Production

Before using Phase 7b in production:

1. **Vault connectivity test:**
   ```bash
   # Test with real Vault
   vault status
   vault kv get secret/path/to/test
   ```

2. **AWS credentials test:**
   ```bash
   aws sts get-caller-identity
   aws secretsmanager list-secrets
   ```

3. **Azure credentials test:**
   ```bash
   az account show
   az keyvault secret list --vault-name myvault
   ```

4. **GCP credentials test:**
   ```bash
   gcloud auth list
   gcloud secrets list
   ```

---

## Future Test Improvements

### Short Term
- [ ] Add `--dry-run` validation for K8s manifests
- [ ] Add Dockerfile syntax validation
- [ ] Add YAML schema validation for Compose files

### Medium Term
- [ ] CI/CD pipeline with Docker-in-Docker for build tests
- [ ] Kind/minikube cluster for K8s integration tests
- [ ] Mock registry for push tests

### Long Term
- [ ] Full end-to-end deployment tests
- [ ] Performance/load testing
- [ ] Security scanning of generated containers

---

## Running Tests

```bash
# All deployment glue tests (103 tests)
swipl tests/glue/test_deployment_glue.pl

# Expected output includes:
# - Phase 6a-d: 62 tests (production ready)
# - Phase 7a: 21 tests (experimental, code generation only)
# - Phase 7b: 20 tests (experimental, code generation only)
```

## Contributing

When adding new features:

1. **Always add unit tests** for configuration and code generation
2. **Document integration test requirements** in this file
3. **Mark experimental features** with `[EXPERIMENTAL]` in code comments
4. **Update this file** with coverage information

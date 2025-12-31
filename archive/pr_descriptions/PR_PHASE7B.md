# Add Phase 7b: Secrets Management [EXPERIMENTAL]

## Summary

Implements Phase 7b of the Cloud & Enterprise feature set: secrets management for HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, and GCP Secret Manager.

## ⚠️ Experimental Status

**This feature is marked EXPERIMENTAL.**

### What's Tested
- ✅ Configuration storage and retrieval
- ✅ CLI command generation (vault, aws, az, gcloud)
- ✅ Kubernetes manifest structure (Secret, ExternalSecret)
- ✅ Shell script structure for environment injection

### What's NOT Tested
- ❌ Vault authentication actually works
- ❌ AWS credentials are valid
- ❌ Azure authentication succeeds
- ❌ GCP service account access works
- ❌ Secrets are actually retrieved

### Before Production Use

1. **Test Vault connectivity:**
   ```bash
   vault status
   vault kv get secret/path/to/test
   ```

2. **Verify AWS credentials:**
   ```bash
   aws sts get-caller-identity
   aws secretsmanager list-secrets
   ```

3. **Confirm Azure authentication:**
   ```bash
   az account show
   az keyvault secret list --vault-name myvault
   ```

4. **Test GCP access:**
   ```bash
   gcloud auth list
   gcloud secrets list
   ```

See `docs/TEST_COVERAGE.md` for detailed coverage information.

---

## New Features

### Secret Source Configuration

```prolog
% Generic secret source registration
:- declare_secret_source(prod_secrets, [
    type(vault),
    url('https://vault.example.com')
]).
```

### HashiCorp Vault Integration

```prolog
% Configure Vault with multiple auth methods
:- declare_vault_config(prod_vault, [
    url('https://vault.example.com'),
    auth_method(token),           % or: approle, kubernetes, aws_iam
    namespace('production')
]).

% For AppRole authentication
:- declare_vault_config(app_vault, [
    url('https://vault.example.com'),
    auth_method(approle),
    role_id_env('VAULT_ROLE_ID'),
    secret_id_env('VAULT_SECRET_ID')
]).

% Read secrets
generate_vault_read(prod_vault, 'secret/data/myapp', [], Command).
% → vault kv get -address=https://vault.example.com secret/data/myapp

% Read specific field
generate_vault_read(prod_vault, 'secret/data/myapp', [field(password)], Command).
% → vault kv get -address=... -field=password secret/data/myapp

% Generate Vault Agent config
generate_vault_agent_config(prod_vault, [
    template('config.tpl', '/app/config.json')
], AgentConfig).
```

### AWS Secrets Manager

```prolog
:- declare_aws_secrets_config(prod_aws, [
    region('us-east-1'),
    profile('production')
]).

% Read secret
generate_aws_secret_read(prod_aws, 'prod/database/credentials', [], Command).
% → aws secretsmanager get-secret-value --secret-id prod/database/credentials --region us-east-1

% Extract specific JSON key
generate_aws_secret_read(prod_aws, 'prod/database', [key(password)], Command).
% → aws secretsmanager get-secret-value ... | jq -r '.SecretString | fromjson | .password'
```

### Azure Key Vault

```prolog
:- declare_azure_keyvault_config(prod_azure, [
    vault_url('https://myvault.vault.azure.net'),
    tenant('tenant-id')
]).

generate_azure_secret_read(prod_azure, 'database-password', [], Command).
% → az keyvault secret show --vault-name myvault --name database-password --query value -o tsv
```

### GCP Secret Manager

```prolog
:- declare_gcp_secrets_config(prod_gcp, [
    project('my-gcp-project'),
    credentials_file('/path/to/service-account.json')
]).

generate_gcp_secret_read(prod_gcp, 'database-password', [], Command).
% → gcloud secrets versions access latest --secret=database-password --project=my-gcp-project

% Specific version
generate_gcp_secret_read(prod_gcp, 'api-key', [version('5')], Command).
% → gcloud secrets versions access 5 --secret=api-key --project=my-gcp-project
```

### Service Secret Bindings

```prolog
% Bind secrets to a service
:- declare_service_secrets(api_service, [
    secret(db_password, [source(prod_vault), path('secret/data/db'), field(password)]),
    secret(api_key, [source(prod_aws), path('prod/api-keys'), key(main_key)])
]).

% Generate shell script for environment injection
generate_secret_env_script(api_service, [], Script).
% → #!/bin/bash
%   export DB_PASSWORD="$(vault kv get ...)"
%   export API_KEY="$(aws secretsmanager get-secret-value ...)"

% Generate Kubernetes Secret manifest
generate_k8s_secret(api_service, [], SecretYaml).
% → apiVersion: v1
%   kind: Secret
%   metadata:
%     name: api-service-secrets
%   stringData:
%     DB_PASSWORD: <placeholder>
%     API_KEY: <placeholder>

% Generate ExternalSecret for external-secrets operator
generate_k8s_external_secret(api_service, [provider(vault)], ExternalSecretYaml).
```

### Unified Secret Access

```prolog
% Resolve secret from any source type
resolve_secret(prod_vault, 'secret/data/myapp', [field(password)], Result).
resolve_secret(prod_aws, 'prod/credentials', [key(api_key)], Result).

% List secrets from a source
list_secrets(prod_vault, [path('secret/metadata/')], Command).
list_secrets(prod_aws, [], Command).
```

## Implementation Details

### Supported Secret Backends

| Backend | Auth Methods | Features |
|---------|-------------|----------|
| HashiCorp Vault | token, approle, kubernetes, aws_iam | KV read, field extraction, agent config |
| AWS Secrets Manager | IAM, profile | Secret read, JSON key extraction |
| Azure Key Vault | CLI auth, service principal | Secret read |
| GCP Secret Manager | Service account, CLI auth | Secret read, versioning |

### Kubernetes Integration

| Resource Type | Description |
|--------------|-------------|
| `Secret` | Standard K8s secret with placeholder values |
| `ExternalSecret` | CRD for external-secrets operator |

## Files Changed

| File | Changes |
|------|---------|
| `src/unifyweaver/glue/deployment_glue.pl` | +698 lines |
| `tests/glue/test_deployment_glue.pl` | +213 lines |
| `docs/TEST_COVERAGE.md` | +107 lines |

**Total: ~1,018 insertions(+)**

## Tests

20 new tests (103 total):

| Group | Tests | Coverage Type |
|-------|-------|---------------|
| Secret Source Config | 2 | Unit (config storage) |
| HashiCorp Vault | 4 | Unit (command generation) |
| AWS Secrets Manager | 3 | Unit (command generation) |
| Azure Key Vault | 2 | Unit (command generation) |
| GCP Secret Manager | 2 | Unit (command generation) |
| Service Secrets | 4 | Unit (manifest/script structure) |
| Unified Secret Access | 3 | Unit (command generation) |

**Note:** All Phase 7b tests are unit tests verifying code generation, NOT integration tests.

## Test Plan

```bash
# Run all tests
swipl tests/glue/test_deployment_glue.pl

# Manual validation (recommended before use)
# 1. Generate a Vault read command and test it
# 2. Generate AWS secret read and verify credentials
# 3. Generate K8s manifests and run: kubectl apply --dry-run=client -f -
```

## Phase 7 Progress

- [x] **Phase 7a**: Container Deployment - **EXPERIMENTAL**
  - Docker/Compose support
  - Kubernetes manifests
  - Registry authentication
- [x] **Phase 7b**: Secrets Management (this PR) - **EXPERIMENTAL**
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault
  - GCP Secret Manager
- [ ] **Phase 7c**: Multi-Region & Cloud Functions
  - Geo-failover
  - Lambda/Cloud Functions

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# MPP (Managed Platform) Deployment

This directory contains Kubernetes manifests for deploying the Template Agent to Red Hat Managed Platform.

## Prerequisites

- Access to Red Hat Managed Platform cluster
- `kubectl` configured with cluster access
- Tenant name for your deployment

## Quick Start

Deploy using the Makefile from the project root:

```bash
# Deploy to MPP
make deploy mpp TENANT=ask-data

# Remove deployment
make undeploy mpp TENANT=ask-data
```

## Configuration

### Required Secrets

Update `secret.yaml` before deploying:

```yaml
stringData:
  POSTGRES_USER: ""                    # PostgreSQL username
  POSTGRES_PASSWORD: ""                # PostgreSQL password
  SESSION_SECRET: ""                   # Generate a secure random key
  LANGFUSE_PUBLIC_KEY: ""              # Optional: Langfuse public key
  LANGFUSE_SECRET_KEY: ""              # Optional: Langfuse secret key
  LANGFUSE_HOST: ""                    # Optional: Langfuse host URL
  GOOGLE_SERVICE_ACCOUNT_FILE: ""      # Optional: Google service account path
  GOOGLE_APPLICATION_CREDENTIALS_CONTENT: ""  # Optional: Google credentials JSON
  SSO_CLIENT_ID: ""                    # Optional: SSO client ID
  SSO_CLIENT_SECRET: ""                # Optional: SSO client secret
  SNOWFLAKE_ACCOUNT: ""                # Optional: Snowflake account
```

### ConfigMap Settings

Configure `configmap.yaml` for your environment:

| Setting | Default | Description |
|---------|---------|-------------|
| `PYTHON_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `USE_INMEMORY_SAVER` | `false` | Use in-memory storage instead of PostgreSQL |
| `LANGFUSE_TRACING_ENVIRONMENT` | `production` | Langfuse environment name |
| `POSTGRES_HOST` | - | PostgreSQL host address |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | - | PostgreSQL database name |
| `SSO_CALLBACK_URL` | - | SSO OAuth callback URL |

### Tenant Configuration

Update `tenant.yaml` with your tenant information:

```yaml
spec:
  tenantId: "ask-data"           # Your tenant ID
  appCode: "ASKD-001"            # Your application code
  costCenter: "12345"            # Your cost center
```

## Architecture

### Storage Modes

**PostgreSQL Storage (Default)**
- `USE_INMEMORY_SAVER: false`
- Requires PostgreSQL configuration
- Persistent conversation history
- Production-ready

**In-Memory Storage**
- `USE_INMEMORY_SAVER: true`
- No database required
- History lost on restart
- Suitable for development/testing only

### Resources Deployed

| Resource | Name | Purpose |
|----------|------|---------|
| **BuildConfig** | template-agent | Build container from source |
| **ImageStream** | template-agent | Store built container images |
| **Deployment** | template-agent | Run agent application |
| **Service** | template-agent | Internal cluster service (port 8443) |
| **Route** | template-agent | External HTTPS access with TLS |
| **ConfigMap** | template-agent-config | Environment configuration |
| **Secret** | template-agent-secrets | Sensitive credentials |
| **Tenant** | ask-data | Multi-tenant configuration |

### Network Configuration

- **Service Port**: 8443 (HTTPS)
- **Route**: HTTPS with re-encrypt termination
- **TLS**: Automatic certificate from platform + internal TLS

## Deployment Process

### Manual Deployment

```bash
# Navigate to deployment directory
cd deployment/mpp

# Update configuration files
# - secret.yaml: Add your secrets
# - configmap.yaml: Update PostgreSQL connection details
# - tenant.yaml: Set your tenant ID

# Apply manifests
kubectl apply -k .

# Verify deployment
kubectl get pods -l app=template-agent
kubectl logs -l app=template-agent --tail=50
```

### Build Process

The BuildConfig will:
1. Clone source from Git repository
2. Build container using Containerfile (Red Hat UBI base)
3. Push to internal ImageStream
4. Trigger deployment rollout

### Verify Deployment

```bash
# Check pod status
kubectl get pods -l app=template-agent

# Check pod logs
kubectl logs -l app=template-agent -f

# Check route
kubectl get route template-agent

# Test health endpoint
ROUTE_URL=$(kubectl get route template-agent -o jsonpath='{.spec.host}')
curl https://${ROUTE_URL}/health
```

Expected health response:
```json
{
  "status": "healthy",
  "service": "template-agent"
}
```

## Troubleshooting

### Pod Fails to Start

Check logs for errors:
```bash
kubectl logs -l app=template-agent --tail=100
```

Common issues:
- Missing required secrets
- Invalid configuration values
- Database connection failures (if not using in-memory storage)
- TLS certificate mounting issues

### Build Failures

Check build logs:
```bash
kubectl logs -f bc/template-agent
```

Common issues:
- Containerfile syntax errors
- Missing dependencies in pyproject.toml
- Build timeout

### Route Not Accessible

Check route configuration:
```bash
kubectl describe route template-agent
```

Verify:
- TLS certificate is valid
- Route is admitted
- Service endpoints are available

### Database Connection Issues

Check PostgreSQL connectivity:
```bash
# From within the pod
kubectl exec -it deployment/template-agent -- bash
python -c "from template_agent.src.settings import settings; print(settings.database_uri)"
```

Verify:
- POSTGRES_HOST is reachable from the pod
- POSTGRES_USER and POSTGRES_PASSWORD are correct
- POSTGRES_DB exists

## Cleanup

Remove all resources:

```bash
# Using Makefile
make undeploy mpp TENANT=ask-data

# Or manually
kubectl delete -k deployment/mpp/
```

## Security Considerations

### Production Deployment Best Practices

1. **Secrets Management**:
   - Use external secrets operator
   - Rotate credentials regularly
   - Never commit secrets to git
   - Use strong random keys for SESSION_SECRET

2. **Database Security**:
   - Use encrypted connections to PostgreSQL
   - Restrict database access to agent pods only
   - Regular backups of conversation history

3. **Network Policies**:
   - Restrict ingress to authorized sources
   - Use network policies for pod isolation
   - Limit egress to required services only

4. **Resource Limits**:
   - Set appropriate CPU/memory limits
   - Configure horizontal pod autoscaling for production
   - Monitor resource usage

5. **Monitoring & Logging**:
   - Set up log aggregation
   - Configure health check monitoring
   - Enable metrics collection
   - Use Langfuse for LLM observability

6. **TLS Configuration**:
   - Verify internal TLS certificates are valid
   - Use re-encrypt termination for end-to-end encryption
   - Monitor certificate expiration

## Support

For issues or questions:
- Check application logs: `kubectl logs -l app=template-agent`
- Review pod events: `kubectl describe pod -l app=template-agent`
- Contact platform support team

# Kubernetes Quickstart

## Local Development (kind)

1. Install [kind](https://kind.sigs.k8s.io/) and [Helm](https://helm.sh/)
2. Create a local cluster:

   ```powershell
   kind create cluster --name trader-stack
   ```

3. Load images (if building locally):

   ```powershell
   kind load docker-image trading-system-api:latest --name trader-stack
   kind load docker-image trading-system-ui:latest --name trader-stack
   ```

4. Install the Helm chart:

   ```powershell
   helm install trader-stack charts/trader-stack --values charts/trader-stack/values.yaml
   ```

5. Check status:

   ```powershell
   kubectl get pods,svc
   ```

## Production Deployment

1. Push images to OCI registry (see GitHub Actions workflow)
2. Update `values.yaml` with the correct image tags
3. Install/upgrade:

   ```powershell
   helm upgrade --install trader-stack charts/trader-stack --values charts/trader-stack/values.yaml
   ```

4. Validate manifests:

   ```powershell
   helm template trader-stack charts/trader-stack | kubeval
   ```

## Health Check

- API: `kubectl port-forward svc/trader-stack-api 5000:5000` then `curl http://localhost:5000/health`
- UI:  `kubectl port-forward svc/trader-stack-ui 8501:8501` then open [http://localhost:8501](http://localhost:8501)

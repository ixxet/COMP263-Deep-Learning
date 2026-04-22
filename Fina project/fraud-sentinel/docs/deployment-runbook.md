# Deployment Runbook

## Prerequisites

- GHCR images published as `ghcr.io/ixxet/fraud-sentinel-api:latest`, `ghcr.io/ixxet/fraud-sentinel-trainer:latest`, and `ghcr.io/ixxet/fraud-sentinel-ui:latest`.
- SOPS-managed Secret named `fraud-sentinel-secrets` with `FRAUD_DATABASE_URL`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `FRAUD_VLLM_API_KEY`.
- Existing Postgres, Qdrant, TEI, and vLLM services reachable by the internal URLs in `k8s/base/configmap.yaml`.
- Prometheus Operator CRDs installed for `ServiceMonitor`.
- NVIDIA device plugin/runtime available so the trainer pod can request `nvidia.com/gpu: "1"`.
- Image pull Secret named `ghcr-ixxet` in the `fraud-sentinel` namespace because GHCR packages are private by default.

## Order Of Operations

1. Build and push the backend and frontend images.
2. Let GitHub Actions build and push the `linux/amd64` API, CUDA trainer, and UI images to GHCR.
3. Create the SOPS Secret from `k8s/base/secret.sops.example.yaml`.
4. Create the `ghcr-ixxet` image pull Secret or make the GHCR packages public.
5. Apply the Talos overlay with Flux or `kustomize build k8s/overlays/talos | kubectl apply -f -`.
6. Wait for `fraud-migrate` to complete.
7. Wait for `fraud-rag-seed` to complete.
8. Add Kaggle credentials, then unsuspend `fraud-trainer` and wait for model artifacts on the PVC.
9. Restart `fraud-api` after training if it started before artifacts existed.
10. Open `fraud-ui` through the internal Cilium/Tailscale path.

## Smoke Test

```bash
kubectl -n fraud-sentinel get pods
kubectl -n fraud-sentinel logs job/fraud-migrate
kubectl -n fraud-sentinel logs job/fraud-rag-seed
kubectl -n fraud-sentinel logs job/fraud-trainer
kubectl -n fraud-sentinel port-forward svc/fraud-api 8000:8000
curl http://localhost:8000/readyz
```

Use the UI review sample to create a high-risk case, then run the agent worker and submit an analyst review.

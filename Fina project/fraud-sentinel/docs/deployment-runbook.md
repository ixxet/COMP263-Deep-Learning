# Deployment Runbook

## Prerequisites

- GHCR images published as `ghcr.io/ixxet/fraud-sentinel-api:latest`, `ghcr.io/ixxet/fraud-sentinel-trainer:latest`, and `ghcr.io/ixxet/fraud-sentinel-ui:latest`.
- SOPS-managed Secret named `fraud-sentinel-secrets` with `FRAUD_DATABASE_URL`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `FRAUD_VLLM_API_KEY`. URL-encode the database username and password if they contain reserved characters such as `:`, `@`, `/`, or `#`.
- Existing Postgres, Qdrant, TEI, and vLLM services reachable by the internal URLs in `k8s/base/configmap.yaml`.
- Prometheus Operator CRDs installed for `ServiceMonitor`.
- NVIDIA device plugin/runtime available so the trainer pod can request `nvidia.com/gpu: "1"`.
- Image pull Secret named `ghcr-ixxet` in the `fraud-sentinel` namespace because GHCR packages are private by default.
- Egress from the `fraud-sentinel` namespace to Cloudflare Tunnel endpoints on TCP/UDP `7844`.

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
10. Open `fraud-ui` through the internal Cilium/Tailscale path or the Cloudflare quick tunnel URL from `fraud-sentinel-tunnel` logs.

## Smoke Test

```bash
./ci/smoke_cluster.sh
```

Before training, the smoke suite expects `/readyz` to report `db_ready=true` and `model_ready=false`. After Kaggle training writes artifacts to the PVC and the API is ready, run the stricter path:

```bash
REQUIRE_MODEL_READY=true ./ci/smoke_cluster.sh
```

Use the UI review sample to create a high-risk case, then run the agent worker and submit an analyst review.

## Cloudflare Tunnel

The Talos overlay deploys `fraud-sentinel-tunnel` with `cloudflare/cloudflared:2026.3.0` as a quick tunnel to `http://fraud-ui.fraud-sentinel.svc.cluster.local:3000`. Get the generated public URL with:

```bash
kubectl -n fraud-sentinel logs deploy/fraud-sentinel-tunnel --tail=200 \
  | grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' \
  | tail -1
```

Quick tunnels are for demos and smoke tests. For a stable hostname, create a remotely-managed Cloudflare Tunnel, put the token in a SOPS secret, and switch `cloudflared` to `tunnel --no-autoupdate --loglevel info --metrics 0.0.0.0:2000 run` with `TUNNEL_TOKEN` from that secret.

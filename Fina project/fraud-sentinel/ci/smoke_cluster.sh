#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-fraud-sentinel}"
API_LOCAL_PORT="${API_LOCAL_PORT:-18000}"
UI_LOCAL_PORT="${UI_LOCAL_PORT:-13000}"
REQUIRE_MODEL_READY="${REQUIRE_MODEL_READY:-false}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

wait_http() {
  local url="$1"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  exit 1
}

need kubectl
need curl
need python3

kubectl -n "$NS" wait --for=condition=complete job/fraud-migrate --timeout=120s
kubectl -n "$NS" wait --for=condition=complete job/fraud-rag-seed --timeout=120s
kubectl -n "$NS" wait --for=condition=available deployment/fraud-ui --timeout=180s
kubectl -n "$NS" wait --for=condition=available deployment/fraud-agent --timeout=180s

trainer_gpu="$(kubectl -n "$NS" get job fraud-trainer -o jsonpath='{.spec.template.spec.containers[0].resources.requests.nvidia\.com/gpu}')"
if [[ "$trainer_gpu" != "1" ]]; then
  echo "fraud-trainer should request one NVIDIA GPU, got: $trainer_gpu" >&2
  exit 1
fi

if [[ "$REQUIRE_MODEL_READY" == "true" ]]; then
  kubectl -n "$NS" wait --for=condition=complete job/fraud-trainer --timeout=10s
  kubectl -n "$NS" wait --for=condition=available deployment/fraud-api --timeout=300s
  api_target="svc/fraud-api"
  e2e_mode=(--require-ready --require-case)
else
  trainer_suspended="$(kubectl -n "$NS" get job fraud-trainer -o jsonpath='{.spec.suspend}')"
  if [[ "$trainer_suspended" != "true" ]]; then
    echo "fraud-trainer should stay suspended until Kaggle credentials are present" >&2
    exit 1
  fi
  api_target="pod/$(kubectl -n "$NS" get pod \
    -l app.kubernetes.io/name=fraud-sentinel,app.kubernetes.io/component=api \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}')"
  e2e_mode=(--allow-not-ready)
fi

kubectl -n "$NS" port-forward "$api_target" "$API_LOCAL_PORT:8000" >/tmp/fraud-api-port-forward.log 2>&1 &
PIDS+=("$!")
wait_http "http://127.0.0.1:$API_LOCAL_PORT/healthz"
python3 "$ROOT_DIR/ci/e2e_api.py" --base-url "http://127.0.0.1:$API_LOCAL_PORT" "${e2e_mode[@]}"

kubectl -n "$NS" port-forward svc/fraud-ui "$UI_LOCAL_PORT:3000" >/tmp/fraud-ui-port-forward.log 2>&1 &
PIDS+=("$!")
wait_http "http://127.0.0.1:$UI_LOCAL_PORT/"

if kubectl -n "$NS" get deployment fraud-sentinel-tunnel >/dev/null 2>&1; then
  kubectl -n "$NS" rollout status deployment/fraud-sentinel-tunnel --timeout=180s
  tunnel_url="$(kubectl -n "$NS" logs deployment/fraud-sentinel-tunnel --tail=200 \
    | grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' \
    | tail -1 || true)"
  if [[ -n "$tunnel_url" ]]; then
    echo "Cloudflare quick tunnel: $tunnel_url"
  fi
fi

echo "cluster smoke passed"

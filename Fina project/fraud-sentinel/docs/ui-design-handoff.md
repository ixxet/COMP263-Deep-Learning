# Fraud Sentinel UI And Product Handoff

This handoff is for a designer or frontend developer improving the Fraud Sentinel web UI. It explains what the platform does, what the current UI exposes, which concepts matter, and where the design is intentionally flexible.

## Product Summary

Fraud Sentinel is a deployment-ready fraud detection and analyst review app for the Talos k3s platform.

The core idea is:

1. A trained PyTorch model scores credit-card transactions.
2. Low-risk predictions are stored for audit.
3. Uncertain or high-risk predictions open a review case.
4. LangGraph can prepare policy-grounded review context.
5. A human analyst records the final review action.

The LLM does not decide fraud. The model score opens the case; the human records the operational decision.

## Current User Surfaces

| Surface | Current route | Purpose |
| --- | --- | --- |
| Transaction scoring | `/` | Score one transaction, upload CSV, see model score, open generated case |
| Case queue | `/cases` | See cases, review backlog, risk mix, high-score cases |
| Case detail | `/cases/{case_id}` | Read case context, inspect transaction, submit approve/escalate/dismiss |
| Mermaid diagrams | `/diagrams/*.mmd` | Expose source diagrams directly from the deployed app |

The UI is functional, but it should be considered a working analyst console, not a final design.

## Technology Used

| Layer | Technology | How It Is Used |
| --- | --- | --- |
| Frontend | SvelteKit | Main dashboard, route rendering, proxy endpoints, deployed as `fraud-ui` |
| Frontend language | TypeScript | API client types and safer transaction/case handling |
| Frontend build | Vite | SvelteKit production build inside the UI image |
| API | FastAPI | Prediction, batch upload, case lookup, review actions, readiness, metrics |
| API runtime | Uvicorn | ASGI server inside the backend container |
| Validation | Pydantic | Request and response schema validation |
| ML framework | PyTorch | Dense classifier plus autoencoder anomaly model |
| Data pipeline | pandas, scikit-learn, joblib | CSV parsing, splits, scaling, artifact persistence |
| Training data | Kaggle credit-card fraud dataset | Real labeled historical fraud dataset with `Time`, `Amount`, `V1`-`V28`, `Class` |
| Agent workflow | LangGraph | Durable case review workflow and human-review gate |
| LLM service | vLLM OpenAI-compatible API | Generates grounded analyst brief text; does not score fraud |
| RAG | TEI embeddings + Qdrant | Retrieves local policy/model-card context for case briefs |
| Database | Postgres | Transactions, predictions, cases, reviews, audit events, agent run records |
| Metrics | Prometheus + ServiceMonitor | API readiness, prediction counts, request metrics, case/agent telemetry |
| Deployment | Kustomize + Flux | Talos overlay, Deployments, Jobs, PVCs, ServiceMonitors |
| Images | GHCR | Stores API, trainer, and UI images |
| Access | Cilium LoadBalancer + Cloudflare quick tunnel | Internal access plus demo URL |

## Important Data Concepts

### Transaction Fields

The model expects the Kaggle credit-card fraud schema:

- `Time`: seconds since the first transaction in the dataset.
- `Amount`: transaction amount.
- `V1` through `V28`: anonymized PCA features.
- `Class`: label from Kaggle, accepted only in batch CSVs and ignored for inference.

Design implication: `V1`-`V28` are not meaningful business fields. The UI should not pretend they map to merchant, cardholder, country, or device. Treat them as model feature inputs.

### Scores

| Score | Meaning |
| --- | --- |
| `risk_score` | Supervised classifier probability-like fraud signal |
| `anomaly_score` | Autoencoder reconstruction-error signal |
| `risk_band` | Route: `low`, `uncertain`, or `high` |

High percentages are not proof of fraud. They mean the transaction is similar to fraud patterns or unusual enough to review.

### Case Statuses

| Status | Meaning |
| --- | --- |
| `audit_closed` | Low risk; no review needed |
| `pending_review` | Case opened and waiting for review |
| `awaiting_human_review` | Agent produced context and paused for human decision |
| `approved` | Analyst confirms the model flag |
| `escalated` | Analyst sends case to deeper investigation |
| `dismissed` | Analyst closes as false alarm or not actionable |

## Prediction Workflow

1. User enters one transaction or loads a sample.
2. UI posts to `POST /api/v1/predict`.
3. SvelteKit proxies to FastAPI `POST /v1/predict`.
4. API validates exact live-prediction fields: `Time`, `Amount`, `V1`-`V28`.
5. API loads the trained model bundle from the model PVC.
6. Scaler transforms features.
7. PyTorch classifier produces `risk_score`.
8. Autoencoder produces `anomaly_score`.
9. Risk policy maps scores to `low`, `uncertain`, or `high`.
10. API writes transaction, prediction, and audit rows to Postgres.
11. If uncertain or high, API creates a fraud case.
12. UI shows scores and links to the case if one exists.

Diagram: `docs/diagrams/prediction-flow.mmd`

## Batch Upload Workflow

1. User uploads CSV.
2. API parses rows with pandas.
3. API enforces the configured batch limit, currently 500 rows.
4. Each row is validated independently.
5. Optional `Class` is ignored.
6. Valid rows are scored and persisted.
7. Invalid rows are counted as rejected.
8. Response includes accepted row count, rejected row count, prediction ids, and case ids.

Design implication: batch upload should feel like an import/report workflow, not a single prediction form. It could benefit from a results table, rejected-row explanations, downloadable output, and filterable opened cases.

Diagram: `docs/diagrams/batch-upload-flow.mmd`

## Case Review Workflow

1. A high or uncertain prediction opens `fraud_cases`.
2. `fraud-agent` loads pending cases.
3. LangGraph runs review gates:
   - load case
   - route by risk
   - retrieve policy context from Qdrant
   - generate an analyst brief through vLLM
   - enforce grounding rules
   - pause for human review
4. Human analyst chooses approve, escalate, or dismiss.
5. API records the review, updates the case status, and writes audit events.

Design implication: the case screen should make the decision feel deliberate. The key interaction is not "click a status button"; it is "record a defensible analyst decision."

Diagrams:

- `docs/diagrams/agent-case-review.mmd`
- `docs/diagrams/human-review-states.mmd`

## Training And Deployment Workflow

1. Kaggle credentials are stored in Kubernetes secret material, not committed.
2. Trainer job reads or downloads `creditcard.csv`.
3. Trainer uses the NVIDIA runtime and requests one GPU.
4. G1 validates schema and split behavior.
5. PyTorch classifier and autoencoder train.
6. G2 checks PR-AUC, recall, precision, F1, confusion matrix, and threshold file.
7. Passing artifacts are written to `fraud-model-artifacts` PVC.
8. API restarts or reloads against the artifact bundle.
9. Strict smoke validates model readiness, prediction, batch upload, review, UI, and tunnel.

Diagrams:

- `docs/diagrams/model-training.mmd`
- `docs/diagrams/gitops-deployment.mmd`
- `docs/diagrams/observability-and-data.mmd`

## Ideal Uses

The strongest demo flows are:

### Single Transaction Demo

Use this when explaining the model and review trigger.

1. Load ordinary sample.
2. Score it.
3. Show audit-only or low-risk behavior.
4. Load review sample.
5. Score it.
6. Open generated case.
7. Record analyst rationale and review action.

### Batch CSV Demo

Use this when showing operational workflow.

1. Download sample CSV.
2. Upload it.
3. Show accepted and rejected rows.
4. Open any generated cases.
5. Explain how raw Kaggle `creditcard.csv` rows can be used.

### Platform Demo

Use this when showing deployment readiness.

1. Open Platform Diagrams on the landing page.
2. Show model training diagram.
3. Show prediction flow diagram.
4. Show agent case-review diagram.
5. Show Prometheus/Grafana readiness if available.

## Design Goals

These are recommended directions, not strict requirements.

### Make The App Feel Like An Analyst Workbench

The user should understand:

- what needs attention now
- why the model flagged something
- what action is available
- what gets recorded for audit

Useful design patterns:

- a compact review queue
- clear status chips
- confidence/risk bars with plain-language labels
- timeline for prediction, agent brief, human review
- "next best action" copy on case detail

### Do Not Over-Center The Raw Form

The 30 input fields are necessary but not pleasant. A designer can reduce their visual weight:

- show only `Time` and `Amount` by default
- place `V1`-`V28` behind an expandable "Model features" section
- keep sample buttons prominent
- show result above or beside the feature payload
- use batch upload as a primary path for realistic use

### Make Batch Upload More Useful

Good next UI improvements:

- show accepted and rejected row table
- show why rows were rejected
- show generated case links
- add CSV output download with prediction scores
- show batch-level risk mix chart

### Make Case Review More Defensible

Good next UI improvements:

- decision panel with approve/escalate/dismiss definitions
- required rationale field near the action buttons
- policy/context citations with source titles
- audit timeline
- difference between model score and analyst decision

### Keep Design Direction Open

The app does not need to keep the current layout, palette, card density, or navigation. The stable constraints are the workflows and contracts:

- `/api/v1/predict` scores one transaction.
- `/api/v1/predict/batch` scores CSV rows.
- `/api/v1/cases` lists cases.
- `/api/v1/cases/{case_id}` returns case detail.
- `/api/v1/cases/{case_id}/review` records human action.

Everything around those workflows can be redesigned.

## Design Guardrails

Avoid these because they would misrepresent the system:

- Do not say the LLM detects fraud.
- Do not call a high score "confirmed fraud."
- Do not treat `V1`-`V28` as business-readable features.
- Do not hide the human review/audit nature of approve/escalate/dismiss.
- Do not make Cloudflare quick tunnel look like the final production access model.

Use these instead:

- "model score"
- "review case"
- "analyst decision"
- "policy context"
- "audit trail"
- "similar to known fraud patterns"

## Useful Links In The Repo

| Topic | Path |
| --- | --- |
| Architecture | `docs/architecture.md` |
| Deployment | `docs/deployment-runbook.md` |
| Diagrams | `docs/diagrams/` |
| Frontend root page | `frontend/src/routes/+page.svelte` |
| Case queue | `frontend/src/routes/cases/+page.svelte` |
| Case detail | `frontend/src/routes/cases/[id]/+page.svelte` |
| Frontend API client | `frontend/src/lib/api.ts` |
| API routes | `backend/fraud_sentinel/api/main.py` |
| Feature schema | `backend/fraud_sentinel/feature_schema.py` |
| Risk policy | `backend/fraud_sentinel/risk.py` |
| Agent graph | `backend/fraud_sentinel/agent/graph.py` |
| Training pipeline | `backend/fraud_sentinel/model/training.py` |

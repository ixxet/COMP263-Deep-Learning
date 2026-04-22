create extension if not exists pgcrypto;

create table if not exists transactions (
    transaction_id uuid primary key default gen_random_uuid(),
    payload jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists predictions (
    prediction_id uuid primary key default gen_random_uuid(),
    transaction_id uuid not null references transactions(transaction_id),
    risk_score double precision not null check (risk_score >= 0 and risk_score <= 1),
    anomaly_score double precision not null check (anomaly_score >= 0 and anomaly_score <= 1),
    risk_band text not null check (risk_band in ('low', 'uncertain', 'high')),
    model_version text not null,
    created_at timestamptz not null default now()
);

create table if not exists fraud_cases (
    case_id uuid primary key default gen_random_uuid(),
    prediction_id uuid not null references predictions(prediction_id),
    status text not null,
    risk_band text not null,
    risk_score double precision not null,
    anomaly_score double precision not null,
    model_version text not null,
    brief text,
    policy_context jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists human_reviews (
    review_id uuid primary key default gen_random_uuid(),
    case_id uuid not null references fraud_cases(case_id),
    action text not null check (action in ('approve', 'escalate', 'dismiss')),
    reviewer text not null,
    rationale text not null,
    created_at timestamptz not null default now()
);

create table if not exists fraud_agent_runs (
    agent_run_id uuid primary key default gen_random_uuid(),
    case_id uuid not null references fraud_cases(case_id),
    status text not null,
    state jsonb not null,
    error text,
    created_at timestamptz not null default now()
);

create table if not exists model_registry (
    model_version text primary key,
    artifact_uri text not null,
    metrics jsonb not null,
    thresholds jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists rag_documents (
    rag_document_id uuid primary key default gen_random_uuid(),
    source text not null,
    title text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists audit_events (
    audit_event_id uuid primary key default gen_random_uuid(),
    entity_type text not null,
    entity_id uuid not null,
    event_type text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_predictions_created_at on predictions(created_at desc);
create index if not exists idx_cases_status_created_at on fraud_cases(status, created_at desc);
create index if not exists idx_audit_entity on audit_events(entity_type, entity_id, created_at);

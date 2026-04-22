create table if not exists fraud_agent_runs (
    agent_run_id uuid primary key default gen_random_uuid(),
    case_id uuid not null references fraud_cases(case_id),
    status text not null,
    state jsonb not null,
    error text,
    created_at timestamptz not null default now()
);

create index if not exists idx_fraud_agent_runs_case_created_at
    on fraud_agent_runs(case_id, created_at desc);

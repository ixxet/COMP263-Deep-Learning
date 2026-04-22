# Architecture

```mermaid
flowchart LR
    UI[SvelteKit UI] --> API[FastAPI fraud-api]
    API --> MODEL[PyTorch artifact bundle]
    API --> PG[(Postgres)]
    API --> CASE[Fraud case]
    CASE --> AGENT[LangGraph fraud-agent]
    AGENT --> TEI[TEI embeddings]
    AGENT --> QDRANT[(Qdrant policy corpus)]
    AGENT --> VLLM[vLLM OpenAI API]
    AGENT --> REVIEW[Human review gate]
    REVIEW --> PG
    API --> PROM[Prometheus metrics]
    AGENT --> PROM
```

The hot prediction path is deterministic and model-driven. Agentic review is triggered only for high-risk or uncertain cases, where durable state, RAG retrieval, grounding checks, and human review add operational value.


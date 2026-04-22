"""LangGraph case-review workflow."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, TypedDict

from fraud_sentinel.agent.grounding import validate_brief
from fraud_sentinel.agent.rag import BriefGenerator, PolicyRetriever, deterministic_brief
from fraud_sentinel.repository import Repository
from fraud_sentinel.settings import Settings


class CaseReviewState(TypedDict, total=False):
    case_id: str
    case: dict[str, Any]
    policy_context: list[dict[str, Any]]
    brief: str
    grounding_ok: bool
    grounding_reasons: list[str]
    grounding_attempts: int
    status: str
    human_decision: dict[str, Any]


def build_graph(repo: Repository, settings: Settings, checkpointer):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    retriever = PolicyRetriever(settings)
    generator = BriefGenerator(settings)

    async def load_case(state: CaseReviewState) -> CaseReviewState:
        case = await repo.get_case(state["case_id"])
        if not case:
            raise KeyError(f"case not found: {state['case_id']}")
        return {"case": case}

    def risk_gate(state: CaseReviewState) -> Literal["retrieve_policy", "finalize"]:
        if state["case"]["risk_band"] == "low":
            return "finalize"
        return "retrieve_policy"

    async def retrieve_policy(state: CaseReviewState) -> CaseReviewState:
        case = state["case"]
        query = (
            f"{case['risk_band']} fraud review risk_score {case['risk_score']} "
            f"anomaly_score {case['anomaly_score']} threshold ethics analyst"
        )
        return {"policy_context": await retriever.retrieve(query)}

    async def generate_brief(state: CaseReviewState) -> CaseReviewState:
        if state.get("grounding_attempts", 0) > 0:
            return {"brief": deterministic_brief(state["case"], state["policy_context"])}
        try:
            brief = await generator.generate(state["case"], state["policy_context"])
        except Exception:
            brief = deterministic_brief(state["case"], state["policy_context"])
        return {"brief": brief}

    def grounding_gate(state: CaseReviewState) -> CaseReviewState:
        result = validate_brief(state.get("brief", ""), len(state.get("policy_context", [])))
        return {
            "grounding_ok": result.ok,
            "grounding_reasons": list(result.reasons),
            "grounding_attempts": state.get("grounding_attempts", 0) + 1,
        }

    def route_grounding(state: CaseReviewState) -> Literal["save_brief_for_review", "generate_brief"]:
        return "save_brief_for_review" if state.get("grounding_ok") else "generate_brief"

    async def save_brief_for_review(state: CaseReviewState) -> CaseReviewState:
        await repo.save_case_brief(
            state["case_id"],
            state["brief"],
            state.get("policy_context", []),
            "awaiting_human_review",
        )
        return {"status": "awaiting_human_review"}

    def human_review(state: CaseReviewState) -> Command[Literal["finalize"]]:
        decision = interrupt(
            {
                "case_id": state["case_id"],
                "risk_band": state["case"]["risk_band"],
                "risk_score": state["case"]["risk_score"],
                "anomaly_score": state["case"]["anomaly_score"],
                "brief": state["brief"],
                "actions": ["approve", "escalate", "dismiss"],
            }
        )
        return Command(goto="finalize", update={"human_decision": decision})

    async def finalize(state: CaseReviewState) -> CaseReviewState:
        status = "pending_review"
        if state.get("human_decision"):
            status = {
                "approve": "approved",
                "escalate": "escalated",
                "dismiss": "dismissed",
            }.get(state["human_decision"].get("action"), "pending_review")
        elif state.get("case", {}).get("risk_band") == "low":
            status = "audit_closed"
        await repo.save_case_brief(
            state["case_id"],
            state.get("brief", "Low-risk case closed without LLM review."),
            state.get("policy_context", []),
            status,
        )
        return {"status": status}

    builder = StateGraph(CaseReviewState)
    builder.add_node("load_case", load_case)
    builder.add_node("retrieve_policy", retrieve_policy)
    builder.add_node("generate_brief", generate_brief)
    builder.add_node("grounding_gate", grounding_gate)
    builder.add_node("save_brief_for_review", save_brief_for_review)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "load_case")
    builder.add_conditional_edges("load_case", risk_gate)
    builder.add_edge("retrieve_policy", "generate_brief")
    builder.add_edge("generate_brief", "grounding_gate")
    builder.add_conditional_edges("grounding_gate", route_grounding)
    builder.add_edge("save_brief_for_review", "human_review")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


class CaseReviewService:
    def __init__(self, repo: Repository, settings: Settings) -> None:
        self.repo = repo
        self.settings = settings
        self.graph = None
        self._checkpointer_context = None
        self._checkpointer = None

    def _graph(self):
        if self.graph is None:
            self.graph = build_graph(self.repo, self.settings, self._make_checkpointer())
        return self.graph

    def _make_checkpointer(self):
        if self._checkpointer is not None:
            return self._checkpointer
        if self.settings.database_url:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver

                self._checkpointer_context = PostgresSaver.from_conn_string(
                    self.settings.database_url
                )
                self._checkpointer = self._checkpointer_context.__enter__()
                self._checkpointer.setup()
                return self._checkpointer
            except Exception as exc:
                raise RuntimeError("Postgres LangGraph checkpointer initialization failed") from exc
        from langgraph.checkpoint.memory import InMemorySaver

        self._checkpointer = InMemorySaver()
        return self._checkpointer

    def close(self) -> None:
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(None, None, None)

    async def start_case(self, case_id: str) -> dict[str, Any]:
        graph = self._graph()
        config = {"configurable": {"thread_id": f"case-{case_id}"}}
        state = await graph.ainvoke({"case_id": case_id}, config=config)
        await self.repo.save_agent_run(case_id, "interrupted" if "__interrupt__" in state else "completed", state)
        return state

    async def resume_case(self, case_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import Command

        graph = self._graph()
        config = {"configurable": {"thread_id": f"case-{case_id}"}}
        state = await graph.ainvoke(Command(resume=decision), config=config)
        await self.repo.save_agent_run(case_id, "completed", state)
        return state


async def process_pending_cases(repo: Repository, settings: Settings, limit: int = 10) -> int:
    service = CaseReviewService(repo, settings)
    processed = 0
    try:
        for case in await repo.pending_cases(limit=limit):
            try:
                await service.start_case(case["case_id"])
                processed += 1
            except Exception as exc:
                await repo.save_agent_run(case["case_id"], "failed", {"case_id": case["case_id"]}, str(exc))
        return processed
    finally:
        service.close()


def run_process_pending(repo: Repository, settings: Settings, limit: int = 10) -> int:
    return asyncio.run(process_pending_cases(repo, settings, limit=limit))

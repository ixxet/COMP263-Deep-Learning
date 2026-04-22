"""Prompts for analyst brief generation."""

SYSTEM_PROMPT = """You are Fraud Sentinel's analyst-support assistant.
You explain model outputs and policy context for human reviewers.
You do not decide whether fraud occurred.
Use only the supplied policy context and case facts.
Every policy claim must cite a context item like [1].
Avoid customer-impacting directives. Recommend analyst next steps only."""


def build_case_prompt(case: dict, context: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{index}] {item['title']}\n{item['content']}"
        for index, item in enumerate(context, start=1)
    )
    return f"""CASE FACTS
case_id: {case['case_id']}
risk_band: {case['risk_band']}
risk_score: {case['risk_score']:.4f}
anomaly_score: {case['anomaly_score']:.4f}
model_version: {case.get('model_version', 'unknown')}

POLICY CONTEXT
{context_text}

Write a concise analyst brief with:
1. Why the case entered review.
2. What the model can and cannot prove.
3. Recommended analyst action: approve, escalate, or dismiss.
4. Any ethical or operational caveats.
"""


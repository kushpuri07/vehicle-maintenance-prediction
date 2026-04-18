"""Chat module for Q&A about the generated report.

This is separate from the main LangGraph workflow — it runs *after* the report
is generated, and answers user questions in context. Uses the same RAG system
for grounded responses.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rag.retriever import search

load_dotenv()

LLM_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3


def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=api_key,
        max_tokens=700,
    )


SYSTEM_PROMPT = """You are a vehicle maintenance assistant answering follow-up
questions about an already-generated assessment report.

YOU HAVE ACCESS TO:
1. The full report that was just generated (findings, actions, risk assessment).
2. Fresh excerpts from the maintenance guideline documents relevant to the user's question.

RULES:
- Answer based on the report and the guideline excerpts. Do NOT invent facts.
- If a question isn't covered by the report or excerpts, say so honestly.
- Keep answers concise (2-5 sentences) unless the user asks for detail.
- Cite source documents inline like: "(per 01_brake_maintenance.md)"
- If the user asks about something unsafe (e.g. "can I skip brake replacement?"),
  emphasize the safety implications from the disclaimer.
- Be friendly but professional — you're a technical advisor, not a salesperson.
"""


def _format_report_context(report: dict) -> str:
    """Compact text summary of the report for the LLM context."""
    lines = [
        f"RISK TIER: {report.get('risk_tier', 'UNKNOWN')}",
        f"ML prediction: {'Maintenance needed' if report.get('prediction') == 1 else 'No maintenance needed'} ({report.get('probability', 0):.0%} confidence)",
        "",
        f"EXECUTIVE SUMMARY: {report.get('executive_summary', '—')}",
        "",
    ]

    findings = report.get("detailed_findings", [])
    if findings:
        lines.append("KEY FINDINGS:")
        for i, f in enumerate(findings, 1):
            lines.append(f"  {i}. {f.get('finding', '')} — {f.get('context', '')} [source: {f.get('source', '')}]")
        lines.append("")

    actions = report.get("action_plan", [])
    if actions:
        lines.append("ACTION PLAN:")
        for i, a in enumerate(actions, 1):
            lines.append(
                f"  {i}. [{a.get('timeline', '')}] {a.get('action', '')} — "
                f"Rationale: {a.get('rationale', '')} [source: {a.get('source', '')}, cost: {a.get('cost', 'Varies')}]"
            )
        lines.append("")

    risk = report.get("risk_assessment", "")
    if risk:
        lines.append(f"RISK IF UNADDRESSED: {risk}")
        lines.append("")

    prev = report.get("preventive_recommendations", [])
    if prev:
        lines.append("PREVENTIVE TIPS:")
        for p in prev:
            lines.append(f"  - {p}")

    return "\n".join(lines)


def answer_question(
    question: str,
    report: dict,
    chat_history: list[dict],
) -> str:
    """Answer a follow-up question about the report.

    Args:
        question:      user's question
        report:        the final_report dict from the agent
        chat_history:  list of {"role": "user"|"assistant", "content": "..."}

    Returns:
        The assistant's reply string.
    """
    # Pull relevant doc chunks for this specific question
    hits = search(question, k=3)
    retrieved = "\n\n".join(
        f"--- Excerpt from {d.metadata.get('source', 'unknown')} ---\n{d.page_content}"
        for d in hits
    )

    # Build message list
    report_ctx = _format_report_context(report)
    sys_content = (
        SYSTEM_PROMPT
        + "\n\n=== ASSESSMENT REPORT CONTEXT ===\n"
        + report_ctx
        + "\n\n=== RELEVANT GUIDELINE EXCERPTS (for this specific question) ===\n"
        + retrieved
    )

    messages = [SystemMessage(content=sys_content)]

    # Include prior chat history
    for msg in chat_history[-6:]:   # last 6 turns
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # The new question
    messages.append(HumanMessage(content=question))

    llm = _get_llm()
    response = llm.invoke(messages)
    return response.content.strip()
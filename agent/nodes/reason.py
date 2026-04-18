"""Node 4: Reason.

Generates a DETAILED structured vehicle assessment using Groq (Llama 3.3 70B).
Produces multiple sections: executive summary, detailed findings, specific
actions with rationale, risk assessment, and preventive recommendations.

All output is grounded in retrieved guideline excerpts.
"""
import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import VehicleState

load_dotenv()

LLM_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2


def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file with "
            "GROQ_API_KEY=your_key_here (get one at https://console.groq.com/keys)"
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=api_key,
        max_tokens=2048,
    )


SYSTEM_PROMPT = """You are writing a DETAILED, professional vehicle maintenance assessment report.

The output is a formal technical report — similar to what a fleet manager would
receive from a service advisor. It should be substantial, specific, and
actionable. Not a one-liner.

You MUST ground every claim in the provided guideline excerpts. If something
isn't in the excerpts, do not fabricate it.

OUTPUT RULES:

1. executive_summary: A substantial paragraph (3-5 sentences, roughly 60-100
   words). State the overall condition, primary concerns, risk classification,
   and what the operator should prioritize. Professional tone, no filler.

2. detailed_findings: At least 2-4 findings. Each finding must include:
   - "finding": the observation (short phrase)
   - "context": 1-2 sentences explaining what this means and why it matters
   - "source": filename of the guideline that supports this finding

3. action_plan: Specific actions (not "include in next service"). Each must include:
   - "timeline": "Immediate" | "Within 1 week" | "Within 2 weeks" | "Within 1 month" | "Next service"
   - "action": specific service (e.g. "Replace front brake pads and inspect rotors")
   - "rationale": 1-2 sentences explaining why THIS action for THIS vehicle
   - "source": filename of supporting guideline
   - "cost": ₹ range if in source, else "Varies"

4. risk_assessment: 2-3 sentences describing what happens if recommended
   actions are ignored. Reference compounding effects (e.g. worn brakes on
   worn tires = extended stopping distance).

5. preventive_recommendations: 3-5 bullet tips for ongoing vehicle care
   based on its current profile (age, usage, history).

6. Respond with VALID JSON ONLY. No markdown fences, no preamble.

JSON schema:
{
  "executive_summary": "60-100 word paragraph",
  "detailed_findings": [
    {
      "finding": "short observation",
      "context": "why this matters, 1-2 sentences",
      "source": "filename.md"
    }
  ],
  "action_plan": [
    {
      "timeline": "Immediate | Within 1 week | Within 2 weeks | Within 1 month | Next service",
      "action": "specific action",
      "rationale": "why this action, 1-2 sentences",
      "source": "filename.md",
      "cost": "₹X,XXX – ₹Y,YYY or Varies"
    }
  ],
  "risk_assessment": "2-3 sentences",
  "preventive_recommendations": ["tip 1", "tip 2", "tip 3"]
}
"""


def _build_user_message(state: VehicleState) -> str:
    f = state["prediction_features"]
    tier = state["risk_tier"]
    reasons = "; ".join(state.get("triage_reasons", []))

    brake  = {0: "New", 1: "Good", 2: "Worn Out"}[f.get("Brake_Condition", 0)]
    tire   = {0: "New", 1: "Good", 2: "Worn Out"}[f.get("Tire_Condition", 0)]
    batt   = {0: "New", 1: "Good", 2: "Weak"}[f.get("Battery_Status", 0)]
    maint  = {0: "Good", 1: "Average", 2: "Poor"}[f.get("Maintenance_History", 0)]

    msg = f"""VEHICLE DATA:
- Brake Condition: {brake}
- Tire Condition: {tire}
- Battery Status: {batt}
- Vehicle Age: {f.get('Vehicle_Age', 0)} years
- Odometer: {f.get('Odometer_Reading', 0):,} km
- Reported Issues: {f.get('Reported_Issues', 0)} (scale 0-5)
- Service History: {f.get('Service_History', 0)} past services
- Accident History: {f.get('Accident_History', 0)} incidents
- Maintenance History: {maint}

ML PREDICTION:
- Needs maintenance: {'YES' if state['prediction'] == 1 else 'NO'}
- Confidence: {state['probability']:.0%}

RISK TIER: {tier}
Triage reasons: {reasons}

RELEVANT GUIDELINE EXCERPTS:
"""

    for i, d in enumerate(state.get("retrieved_docs", []), 1):
        msg += f"\n--- Excerpt {i} (from {d['source']}) ---\n{d['content']}\n"

    msg += "\nProduce the DETAILED JSON assessment now. Each section should be substantive."
    return msg


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM output, tolerating minor wrapping."""
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not parse JSON from LLM response:\n{text}")


def reason_node(state: VehicleState) -> VehicleState:
    """Call the LLM to generate the full detailed report sections."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(state)),
    ]
    response = llm.invoke(messages)
    raw = response.content

    try:
        parsed = _parse_json(raw)
        return {
            "executive_summary":          parsed.get("executive_summary", ""),
            "health_summary":             parsed.get("executive_summary", ""),  # backwards compat
            "detailed_findings":          parsed.get("detailed_findings", []),
            "action_plan":                parsed.get("action_plan", []),
            "risk_assessment":            parsed.get("risk_assessment", ""),
            "preventive_recommendations": parsed.get("preventive_recommendations", []),
        }
    except Exception as e:
        return {
            "executive_summary": f"(LLM output could not be parsed - {e})",
            "health_summary":    f"(LLM output could not be parsed - {e})",
            "detailed_findings": [],
            "action_plan":       [],
            "risk_assessment":   "",
            "preventive_recommendations": [],
            "error": f"reason_node parse error: {e}",
        }
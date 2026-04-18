"""Node 5: Report.

Assembles the final structured report for the UI.
"""
from agent.state import VehicleState


SAFETY_DISCLAIMER = (
    "This maintenance assessment is a decision-support tool based on the data "
    "provided. It does not replace on-site inspection by a certified automotive "
    "technician. Safety-critical services (brakes, steering, tires, airbags) must "
    "be verified and performed by qualified professionals. The operator assumes "
    "full responsibility for servicing decisions."
)


def report_node(state: VehicleState) -> VehicleState:
    """Assemble the final report dict that the UI will render."""
    tier = state.get("risk_tier", "UNKNOWN")

    cited = set()
    for item in state.get("action_plan", []):
        if item.get("source"):
            cited.add(item["source"])
    for item in state.get("detailed_findings", []):
        if item.get("source"):
            cited.add(item["source"])

    report = {
        "vehicle_id":                 state.get("vehicle_id", "—"),
        "risk_tier":                  tier,
        "prediction":                 state.get("prediction"),
        "probability":                state.get("probability"),
        "triage_reasons":             state.get("triage_reasons", []),
        "executive_summary":          state.get("executive_summary", ""),
        "health_summary":             state.get("health_summary", ""),
        "detailed_findings":          state.get("detailed_findings", []),
        "action_plan":                state.get("action_plan", []),
        "risk_assessment":            state.get("risk_assessment", ""),
        "preventive_recommendations": state.get("preventive_recommendations", []),
        "sources_cited":              sorted(cited),
        "disclaimer":                 SAFETY_DISCLAIMER,
        "vehicle_features":           state.get("prediction_features", {}),
    }

    return {"final_report": report}
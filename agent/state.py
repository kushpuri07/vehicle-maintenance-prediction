"""State object passed between LangGraph nodes."""
from typing import TypedDict, Optional, List


class VehicleState(TypedDict, total=False):
    # Input
    vehicle_input: dict
    vehicle_id: str

    # Added by Predict
    prediction: int
    probability: float
    prediction_features: dict

    # Added by Triage 
    risk_tier: str
    triage_reasons: List[str]

    # Added by Retrieve
    retrieved_docs: List[dict]
    retrieval_query: str

    # Added by Reason (now much richer) 
    executive_summary: str                          # detailed paragraph
    detailed_findings: List[dict]                   # [{finding, context, source}]
    action_plan: List[dict]                         # [{timeline, action, rationale, source, cost}]
    risk_assessment: str                            # what happens if unaddressed
    preventive_recommendations: List[str]           # proactive tips
    health_summary: str                             # kept for backwards compat (short version)
    # Added by Report 
    final_report: dict
    # Misc 
    error: Optional[str]
"""State object passed between LangGraph nodes.

Each node reads what it needs and adds its own output. By the end,
the state holds the complete picture of the vehicle analysis.
"""
from typing import TypedDict, Optional, List


class VehicleState(TypedDict, total=False):
    # --- Input (provided by the UI) ---
    vehicle_input: dict          # raw user input from form/CSV row
    vehicle_id: str              # identifier (for fleet mode)

    # --- Added by Predict node ---
    prediction: int              # 0 or 1
    probability: float           # 0.0 to 1.0
    prediction_features: dict    # the feature values used

    # --- Added by Triage node ---
    risk_tier: str               # CRITICAL / HIGH / MEDIUM / LOW
    triage_reasons: List[str]    # why this tier was assigned

    # --- Added by Retrieve node ---
    retrieved_docs: List[dict]   # [{source, content}, ...]
    retrieval_query: str         # the query that was used

    # --- Added by Reason node ---
    health_summary: str          # LLM's summary of the situation
    action_plan: List[dict]      # [{timeline, action, source, cost}, ...]

    # --- Added by Report node ---
    final_report: dict           # fully structured output for UI

    # --- Misc ---
    error: Optional[str]

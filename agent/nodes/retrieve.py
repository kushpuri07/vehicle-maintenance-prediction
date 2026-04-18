"""Node 3: Retrieve.

Builds a query describing what's wrong with the vehicle and pulls the most
relevant chunks from the maintenance docs via RAG.

The query is built deterministically from the feature values so we always
retrieve docs relevant to the actual problems (not just a generic query).
"""
from agent.state import VehicleState
from rag.retriever import search


# Reverse maps for turning encoded integers back into words for the query
TIRE_BRAKE = {0: "new", 1: "good", 2: "worn out"}
BATTERY    = {0: "new", 1: "good", 2: "weak"}
MAINT      = {0: "good maintenance", 1: "average maintenance", 2: "poor maintenance"}


def _build_query(f: dict, tier: str) -> str:
    """Build a descriptive query from the vehicle's condition."""
    parts = []

    # Mention worst components first
    if f.get("Brake_Condition", 0) == 2:
        parts.append("worn out brakes")
    if f.get("Tire_Condition", 0) == 2:
        parts.append("worn out tires")
    if f.get("Battery_Status", 0) == 2:
        parts.append("weak battery")

    # Mention reported issues if notable
    issues = f.get("Reported_Issues", 0)
    if issues >= 3:
        parts.append(f"{issues} reported issues")

    # Mention age if vehicle is old
    age = f.get("Vehicle_Age", 0)
    if age >= 6:
        parts.append(f"{age} year old vehicle")

    # Mention maintenance history if not good
    if f.get("Maintenance_History", 0) >= 1:
        parts.append(MAINT[f["Maintenance_History"]])

    # Mention accidents if any
    accid = f.get("Accident_History", 0)
    if accid >= 2:
        parts.append(f"{accid} past accidents")

    # Fallback: healthy vehicle query
    if not parts:
        parts.append("healthy vehicle routine maintenance schedule")

    # Always add tier context so the retriever pulls the right priority level
    return f"{tier.lower()} risk vehicle: " + ", ".join(parts)


def retrieve_node(state: VehicleState) -> VehicleState:
    """Run RAG query and attach retrieved context to state."""
    f = state["prediction_features"]
    tier = state["risk_tier"]

    query = _build_query(f, tier)
    docs = search(query, k=5)

    retrieved = [
        {"source": d.metadata.get("source", "unknown"),
         "content": d.page_content}
        for d in docs
    ]

    return {
        "retrieval_query": query,
        "retrieved_docs": retrieved,
    }

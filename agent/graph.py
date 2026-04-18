"""LangGraph workflow that wires the 5 nodes together.

Flow: predict -> triage -> retrieve -> reason -> report -> END

Usage:
    from agent.graph import run_agent
    result = run_agent({
        'Brake_Condition': 2, 'Tire_Condition': 2, 'Vehicle_Age': 9,
        'Battery_Status': 2, 'Reported_Issues': 4, 'Service_History': 2,
        'Odometer_Reading': 135000, 'Insurance_Premium': 25000,
        'Accident_History': 2, 'Maintenance_History': 2,
    })
    print(result['final_report'])
"""
from langgraph.graph import StateGraph, END

from agent.state import VehicleState
from agent.nodes.predict  import predict_node
from agent.nodes.triage   import triage_node
from agent.nodes.retrieve import retrieve_node
from agent.nodes.reason   import reason_node
from agent.nodes.report   import report_node


def build_graph():
    """Compile the LangGraph workflow."""
    g = StateGraph(VehicleState)

    g.add_node("predict",  predict_node)
    g.add_node("triage",   triage_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason",   reason_node)
    g.add_node("report",   report_node)

    g.set_entry_point("predict")
    g.add_edge("predict",  "triage")
    g.add_edge("triage",   "retrieve")
    g.add_edge("retrieve", "reason")
    g.add_edge("reason",   "report")
    g.add_edge("report",   END)

    return g.compile()


# Compile once at import time
_GRAPH = None

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(vehicle_input: dict, vehicle_id: str = "Vehicle-1") -> dict:
    """Run the full agent pipeline on a single vehicle's data.

    Returns the final state dict including `final_report`.
    """
    graph = get_graph()
    initial_state: VehicleState = {
        "vehicle_input": vehicle_input,
        "vehicle_id":    vehicle_id,
    }
    return graph.invoke(initial_state)


# Quick test when run directly: python -m agent.graph
if __name__ == "__main__":
    import json

    test_vehicle = {
        "Brake_Condition":     2,   # Worn Out
        "Tire_Condition":      2,   # Worn Out
        "Vehicle_Age":         9,
        "Battery_Status":      2,   # Weak
        "Reported_Issues":     4,
        "Service_History":     2,
        "Odometer_Reading": 135000,
        "Insurance_Premium": 25000,
        "Accident_History":    2,
        "Maintenance_History": 2,   # Poor
    }

    print("Running agent on test vehicle (should be CRITICAL)...\n")
    result = run_agent(test_vehicle, vehicle_id="TEST-001")

    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(json.dumps(result["final_report"], indent=2, ensure_ascii=False))

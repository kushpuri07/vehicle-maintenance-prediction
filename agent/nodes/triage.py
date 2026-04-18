"""Node 2: Triage.

Deterministic rule-based classification into CRITICAL / HIGH / MEDIUM / LOW
following the tier definitions in maintenance_docs/07_fleet_risk_tiers.md.

Rules are hard-coded (not LLM) so the safety-critical classification is
reproducible and can never be hallucinated away.
"""
from agent.state import VehicleState

# Encoding reminders (must match Cell 5 of notebook):
#   Brake_Condition / Tire_Condition:  New=0, Good=1, Worn Out=2
#   Battery_Status:                    New=0, Good=1, Weak=2
#   Maintenance_History:               Good=0, Average=1, Poor=2


def _count_worst_components(f: dict) -> int:
    """How many of the 3 wear components are in worst state?"""
    return sum([
        f.get("Brake_Condition", 0)  == 2,
        f.get("Tire_Condition", 0)   == 2,
        f.get("Battery_Status", 0)   == 2,
    ])


def triage_node(state: VehicleState) -> VehicleState:
    """Assign a risk tier with explicit reasoning."""
    f = state["prediction_features"]
    proba = state["probability"]
    reasons = []

    brake  = f.get("Brake_Condition", 0)
    tire   = f.get("Tire_Condition", 0)
    batt   = f.get("Battery_Status", 0)
    issues = f.get("Reported_Issues", 0)
    age    = f.get("Vehicle_Age", 0)
    maint  = f.get("Maintenance_History", 0)
    accid  = f.get("Accident_History", 0)

    worst_count = _count_worst_components(f)

    # ─── CRITICAL ─────────────────────────────────────────────
    if brake == 2:
        reasons.append("Brake condition is Worn Out — immediate safety risk")
    if issues >= 5:
        reasons.append(f"Reported issues at maximum ({issues})")
    if worst_count >= 2:
        reasons.append(f"{worst_count} components simultaneously in worst state")
    if state["prediction"] == 1 and proba >= 0.95 and worst_count >= 1:
        reasons.append(f"Model predicts maintenance with {proba:.0%} confidence + worn component")

    if reasons:
        return {"risk_tier": "CRITICAL", "triage_reasons": reasons}

    # ─── HIGH ─────────────────────────────────────────────────
    if worst_count == 1:
        if tire == 2: reasons.append("Tire condition is Worn Out")
        if batt == 2: reasons.append("Battery is Weak")
    if issues in (3, 4):
        reasons.append(f"Elevated reported issue count ({issues})")
    if proba >= 0.80 and proba < 0.95:
        reasons.append(f"Model predicts maintenance with {proba:.0%} confidence")
    if age >= 10 and maint == 2:
        reasons.append("10+ year old vehicle with Poor maintenance history")
    if accid >= 3:
        reasons.append(f"High accident history ({accid} incidents)")

    if reasons:
        return {"risk_tier": "HIGH", "triage_reasons": reasons}

    # ─── MEDIUM ───────────────────────────────────────────────
    if issues == 2:
        reasons.append("2 reported issues")
    if maint >= 1:
        reasons.append("Maintenance history is Average or Poor")
    if 6 <= age <= 9:
        reasons.append(f"Aging vehicle ({age} years old)")
    if 0.60 <= proba < 0.80:
        reasons.append(f"Model predicts maintenance with {proba:.0%} confidence")

    if reasons:
        return {"risk_tier": "MEDIUM", "triage_reasons": reasons}

    # ─── LOW ──────────────────────────────────────────────────
    return {
        "risk_tier": "LOW",
        "triage_reasons": ["All indicators within healthy range"],
    }

"""Streamlit UI for the Vehicle Maintenance & Fleet Management Agent.

Chatbot is now in the sidebar — toggleable, non-intrusive, and always
accessible once a report is generated.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import io
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from agent.graph import run_agent
from agent.nodes.predict import FEATURE_ORDER
from agent.chat import answer_question


# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Fleet Management Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for polished look ───────────────────────────────────
st.markdown(
    """
    <style>
    /* Style the sidebar into a chat panel */
    section[data-testid="stSidebar"] {
        background: #0f0f0f;
        border-left: 1px solid #333;
        width: 420px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Label the sidebar toggle arrow with "Ask AI" text */
    /* When sidebar is collapsed */
    button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] {
        position: relative !important;
    }
    [data-testid="stSidebarCollapsedControl"]::after,
    [data-testid="collapsedControl"]::after {
        content: "Ask AI";
        position: absolute;
        left: 44px;
        top: 50%;
        transform: translateY(-50%);
        background: linear-gradient(135deg, #FF1744, #FF9100);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        white-space: nowrap;
        box-shadow: 0 2px 12px rgba(255, 23, 68, 0.4);
        pointer-events: none;
        animation: pulse-ai 2s infinite;
    }
    @keyframes pulse-ai {
        0%, 100% { transform: translateY(-50%) scale(1); opacity: 1; }
        50%      { transform: translateY(-50%) scale(1.05); opacity: 0.9; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Vehicle Fleet Management Assistant")
st.caption(
    "AI-powered maintenance analysis — ML prediction + RAG-grounded action plans. "
    "Milestone 2: Agentic Fleet Management."
)


# ── Constants ──────────────────────────────────────────────────────
TIER_COLORS = {
    "CRITICAL": "#FF1744",
    "HIGH":     "#FF9100",
    "MEDIUM":   "#FFC400",
    "LOW":      "#00C853",
}

TIMELINE_ORDER = [
    "Immediate", "Within 1 week", "Within 2 weeks", "Within 1 month", "Next service",
]

LABELS = {
    "Brake_Condition":     {0: "New", 1: "Good", 2: "Worn Out"},
    "Tire_Condition":      {0: "New", 1: "Good", 2: "Worn Out"},
    "Battery_Status":      {0: "New", 1: "Good", 2: "Weak"},
    "Maintenance_History": {0: "Good", 1: "Average", 2: "Poor"},
}


# ── Radar Chart ────────────────────────────────────────────────────
def render_health_radar(features: dict, size: float = 5.0):
    scores = {
        "Brakes":       {0: 100, 1: 70, 2: 15}[features.get("Brake_Condition", 1)],
        "Tires":        {0: 100, 1: 70, 2: 15}[features.get("Tire_Condition", 1)],
        "Battery":      {0: 100, 1: 70, 2: 20}[features.get("Battery_Status", 1)],
        "Age":          max(0, 100 - features.get("Vehicle_Age", 0) * 7),
        "Reports":      max(0, 100 - features.get("Reported_Issues", 0) * 20),
        "Maintenance":  {0: 100, 1: 55, 2: 20}[features.get("Maintenance_History", 1)],
        "Accidents":    max(0, 100 - features.get("Accident_History", 0) * 30),
    }

    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(size, size), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    avg = sum(scores.values()) / len(scores)
    if   avg >= 75: fill_color = "#00C853"
    elif avg >= 50: fill_color = "#FFC400"
    elif avg >= 30: fill_color = "#FF9100"
    else:           fill_color = "#FF1744"

    ax.plot(angles, values, color=fill_color, linewidth=2)
    ax.fill(angles, values, color=fill_color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([], color='#888', fontsize=7)
    ax.grid(color='#333')
    ax.spines['polar'].set_color('#333')
    ax.set_title(f"Health Score: {avg:.0f}/100",
                 color='white', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig, avg


# ── Data Sheet ─────────────────────────────────────────────────────
def render_data_sheet(features: dict):
    def _val(key, fmt=None):
        raw = features.get(key, "—")
        if key in LABELS:
            return LABELS[key].get(raw, str(raw))
        if fmt == "km":  return f"{raw:,} km"
        if fmt == "inr": return f"₹{raw:,}"
        return str(raw)

    rows = [
        ("Brake Condition",     _val("Brake_Condition")),
        ("Tire Condition",      _val("Tire_Condition")),
        ("Battery Status",      _val("Battery_Status")),
        ("Maintenance History", _val("Maintenance_History")),
        ("Vehicle Age",         f"{features.get('Vehicle_Age', 0)} years"),
        ("Odometer Reading",    _val("Odometer_Reading", "km")),
        ("Reported Issues",     f"{features.get('Reported_Issues', 0)} / 5"),
        ("Service History",     f"{features.get('Service_History', 0)} past services"),
        ("Accident History",    f"{features.get('Accident_History', 0)} incidents"),
        ("Insurance Premium",   _val("Insurance_Premium", "inr")),
    ]

    mid = (len(rows) + 1) // 2
    left_rows, right_rows = rows[:mid], rows[mid:]

    def row_html(label, value):
        return (
            f"<tr>"
            f"<td style='padding:8px 14px; color:#888; font-size:13px; border-bottom:1px solid #222;'>{label}</td>"
            f"<td style='padding:8px 14px; color:#fff; font-size:14px; border-bottom:1px solid #222; font-weight:500;'>{value}</td>"
            f"</tr>"
        )

    left_html  = "".join(row_html(l, v) for l, v in left_rows)
    right_html = "".join(row_html(l, v) for l, v in right_rows)

    st.markdown(
        f"""
        <div style="background:#1a1a1a; border-radius:6px; padding:12px; margin-bottom:24px;">
          <table style="width:48%; display:inline-table; vertical-align:top; border-collapse:collapse;">
            {left_html}
          </table>
          <table style="width:48%; display:inline-table; vertical-align:top; border-collapse:collapse; margin-left:3%;">
            {right_html}
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_box(content_html, border_color="#333"):
    st.markdown(
        f"""<div style="padding:18px 22px; background:#1a1a1a;
                       border-left:4px solid {border_color}; border-radius:6px;
                       color:#e8e8e8; font-size:15px; line-height:1.7;
                       margin-bottom:24px;">
            {content_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Report Rendering ───────────────────────────────────────────────
def render_report(report: dict, features: dict | None = None):
    tier  = report.get("risk_tier", "UNKNOWN")
    color = TIER_COLORS.get(tier, "#888")
    prob  = report.get("probability", 0)
    prediction = report.get("prediction", 0)
    vehicle_id = report.get("vehicle_id", "—")

    if prediction == 1:
        ml_verdict = "Maintenance Required"
        ml_confidence = prob
    else:
        ml_verdict = "No Maintenance Required"
        ml_confidence = 1 - prob

    if features is None:
        features = report.get("vehicle_features", {})

    st.markdown(
        f"""
        <div style="background:#0f0f0f; border:1px solid #333;
                    border-radius:8px; padding:24px 28px; margin-bottom:24px;">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;
                      border-bottom:1px solid #333; padding-bottom:16px; margin-bottom:16px;">
            <div>
              <div style="font-size:13px; color:#888; letter-spacing:1.5px;">
                VEHICLE MAINTENANCE ASSESSMENT REPORT
              </div>
              <div style="font-size:22px; color:#fff; font-weight:bold; margin-top:6px;">
                Report ID: {vehicle_id}
              </div>
              <div style="font-size:13px; color:#888; margin-top:2px;">
                Assessment Date: {datetime.now().strftime('%B %d, %Y')}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:12px; color:#888; letter-spacing:1.5px;">RISK CLASSIFICATION</div>
              <div style="font-size:32px; font-weight:bold; color:{color}; margin-top:4px; line-height:1;">
                {tier}
              </div>
            </div>
          </div>
          <div style="color:#ccc; font-size:14px;">
            <b>ML Prediction:</b> {ml_verdict}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Model Confidence:</b> {ml_confidence:.1%}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 1. Executive Summary")
    summary = report.get("executive_summary") or report.get("health_summary") or "—"
    section_box(summary, border_color=color)

    if features:
        st.markdown("#### 2. Vehicle Data Sheet")
        render_data_sheet(features)

    section_num = 3 if features else 2
    if features:
        st.markdown(f"#### {section_num}. Vehicle Health Overview")
        col_pad_l, col_chart, col_pad_r = st.columns([1, 2, 1])
        with col_chart:
            fig, _ = render_health_radar(features, size=5.0)
            st.pyplot(fig, use_container_width=True)
        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
        section_num += 1

    st.markdown(f"#### {section_num}. Key Findings")
    findings = report.get("detailed_findings", [])
    triage_reasons = report.get("triage_reasons", [])

    if findings:
        for i, f in enumerate(findings, 1):
            finding_text = f.get("finding", "—")
            context      = f.get("context", "")
            src          = f.get("source", "")
            src_line = (f"<div style='margin-top:8px; font-size:11px; color:#888;'>"
                        f"Reference: <code>{src}</code></div>") if src else ""
            st.markdown(
                f"""
                <div style="padding:16px 22px; background:#1a1a1a; border-radius:6px;
                            border-left:3px solid #666; margin-bottom:12px;">
                  <div style="font-size:15px; color:#fff; font-weight:600; margin-bottom:6px;">
                    {section_num}.{i}  {finding_text}
                  </div>
                  <div style="font-size:14px; color:#c8c8c8; line-height:1.6;">
                    {context}
                  </div>
                  {src_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif triage_reasons:
        bullets = "".join(f"<li style='margin-bottom:10px;'>{r}</li>" for r in triage_reasons)
        st.markdown(
            f"""<div style="padding:16px 24px 16px 40px; background:#1a1a1a;
                           border-radius:6px; margin-bottom:24px;">
                <ul style="margin:0; color:#e0e0e0; font-size:15px;">{bullets}</ul>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No material findings.")
    section_num += 1

    st.markdown(f"#### {section_num}. Recommended Actions")
    plan = report.get("action_plan", [])

    if not plan:
        section_box("No specific actions required. Continue manufacturer-scheduled maintenance.")
    else:
        def _tl_rank(item):
            tl = item.get("timeline", "")
            for i, t in enumerate(TIMELINE_ORDER):
                if t.lower() in tl.lower():
                    return i
            return len(TIMELINE_ORDER)

        for idx, item in enumerate(sorted(plan, key=_tl_rank), 1):
            timeline  = item.get("timeline", "—")
            action    = item.get("action", "—")
            rationale = item.get("rationale", "")
            cost      = item.get("cost", "Varies")
            src       = item.get("source", "")

            if   "immediate" in timeline.lower(): accent = "#FF1744"
            elif "week" in timeline.lower():      accent = "#FF9100"
            elif "month" in timeline.lower():     accent = "#FFC400"
            else:                                  accent = "#00C853"

            rat_block = (
                f"<div style='margin-top:10px; padding:10px 14px; background:#0f0f0f;"
                f" border-radius:4px; font-size:13px; color:#bbb; line-height:1.6;'>"
                f"<b style='color:#ddd;'>Rationale:</b> {rationale}</div>"
            ) if rationale else ""
            src_line = (f"<div style='margin-top:8px; font-size:11px; color:#888;'>"
                        f"Reference: <code>{src}</code></div>") if src else ""

            st.markdown(
                f"""
                <div style="padding:16px 22px; border-radius:6px;
                            background:#1a1a1a; border-left:4px solid {accent};
                            margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <span style="color:{accent}; font-weight:bold; font-size:12px; letter-spacing:1.5px;">
                      {section_num}.{idx}  {timeline.upper()}
                    </span>
                    <span style="color:#888; font-size:13px;">Cost: {cost}</span>
                  </div>
                  <div style="margin-top:8px; font-size:16px; color:#fff; font-weight:500; line-height:1.5;">
                    {action}
                  </div>
                  {rat_block}
                  {src_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
    section_num += 1

    risk = report.get("risk_assessment", "")
    if risk:
        st.markdown(f"#### {section_num}. Risk if Unaddressed")
        section_box(risk, border_color="#FF9100")
        section_num += 1

    prev = report.get("preventive_recommendations", [])
    if prev:
        st.markdown(f"#### {section_num}. Preventive Recommendations")
        bullets = "".join(f"<li style='margin-bottom:10px;'>{p}</li>" for p in prev)
        st.markdown(
            f"""<div style="padding:16px 24px 16px 40px; background:#1a1a1a;
                           border-radius:6px; margin-bottom:24px;">
                <ul style="margin:0; color:#e0e0e0; font-size:15px; line-height:1.7;">{bullets}</ul>
            </div>""",
            unsafe_allow_html=True,
        )
        section_num += 1

    sources = report.get("sources_cited", [])
    if sources:
        st.markdown(f"#### {section_num}. Sources Consulted")
        src_lines = "".join(f"<li style='margin-bottom:6px;'><code>{s}</code></li>" for s in sources)
        st.markdown(
            f"""<div style="padding:14px 24px 14px 40px; background:#1a1a1a;
                           border-radius:6px; color:#ccc; font-size:14px;
                           margin-bottom:20px;">
                <ul style="margin:0;">{src_lines}</ul>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div style="margin-top:16px; padding:14px 18px; background:#0f0f0f;
                       border:1px dashed #444; border-radius:6px;
                       font-size:12px; color:#888; line-height:1.6;">
            <b>DISCLAIMER:</b> {report.get("disclaimer", "")}
        </div>""",
        unsafe_allow_html=True,
    )


def render_agent_trace(result: dict):
    with st.expander("Agent reasoning trace (how the decision was made)"):
        st.markdown("**Step 1 — ML Prediction**")
        st.json({
            "prediction":  result.get("prediction"),
            "probability": f"{result.get('probability', 0):.4f}",
        })
        st.markdown("**Step 2 — Triage (rule-based)**")
        st.json({
            "risk_tier":      result.get("risk_tier"),
            "triage_reasons": result.get("triage_reasons", []),
        })
        st.markdown("**Step 3 — RAG Retrieval**")
        st.write(f"Query: *{result.get('retrieval_query', '—')}*")
        for i, d in enumerate(result.get("retrieved_docs", []), 1):
            st.caption(f"[{i}] {d['source']}")
            st.text(d["content"][:250] + ("..." if len(d["content"]) > 250 else ""))
        st.markdown("**Step 4 — LLM Reasoning → Step 5 — Final Report**")
        st.caption("See report above.")


# ── Sidebar Chatbot ────────────────────────────────────────────────
def render_sidebar_chatbot():
    """Chatbot lives in the sidebar — toggleable, non-intrusive."""
    if "current_report" not in st.session_state:
        with st.sidebar:
            st.markdown("### Assistant Chat")
            st.caption("Generate a report first, then ask follow-up questions here.")
        return

    with st.sidebar:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#1a1a1a,#0f0f0f);
                        padding:16px 18px; border-radius:8px;
                        border:1px solid #333; margin-bottom:12px;">
              <div style="font-size:16px; font-weight:bold; color:#fff;">
                Report Assistant
              </div>
              <div style="font-size:12px; color:#888; margin-top:4px;">
                Ask anything about the current report.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested prompt buttons
        st.caption("Quick questions:")
        suggested = [
            "What should I prioritize first?",
            "Why is this the risk tier?",
            "What if I delay the service?",
            "Explain the costs.",
        ]
        for i, q in enumerate(suggested):
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.markdown("---")

        # Chat history
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input handling
        pending = st.session_state.pop("pending_question", None)
        user_input = st.chat_input("Type a question...")
        question = pending or user_input

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(
                        question=question,
                        report=st.session_state["current_report"],
                        chat_history=st.session_state.get("chat_history", []),
                    )
                st.markdown(answer)

            st.session_state.setdefault("chat_history", []).append(
                {"role": "user", "content": question}
            )
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )

        if st.session_state.get("chat_history"):
            if st.button("Clear chat", key="clear_chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()


# ── Form ───────────────────────────────────────────────────────────
def build_single_vehicle_input():
    col1, col2 = st.columns(2)
    with col1:
        brake = st.selectbox("Brake Condition", [0, 1, 2],
            format_func=lambda x: {0: "New", 1: "Good", 2: "Worn Out"}[x], index=1)
        tire = st.selectbox("Tire Condition", [0, 1, 2],
            format_func=lambda x: {0: "New", 1: "Good", 2: "Worn Out"}[x], index=1)
        battery = st.selectbox("Battery Status", [0, 1, 2],
            format_func=lambda x: {0: "New", 1: "Good", 2: "Weak"}[x], index=1)
        maintenance = st.selectbox("Maintenance History", [0, 1, 2],
            format_func=lambda x: {0: "Good", 1: "Average", 2: "Poor"}[x], index=1)
        age = st.slider("Vehicle Age (years)", 0, 15, 5)
    with col2:
        issues = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
        service = st.slider("Service History (# past services)", 0, 12, 4)
        accident = st.selectbox("Accident History", [0, 1, 2, 3], index=1)
        odo = st.number_input("Odometer Reading (km)", 0, 400_000, 60_000, step=1_000)
        premium = st.number_input("Insurance Premium (INR)", 1_500, 100_000, 15_000, step=500)

    return {
        "Brake_Condition": brake, "Tire_Condition": tire, "Vehicle_Age": age,
        "Battery_Status": battery, "Reported_Issues": issues, "Service_History": service,
        "Odometer_Reading": odo, "Insurance_Premium": premium,
        "Accident_History": accident, "Maintenance_History": maintenance,
    }


# ── Tabs ───────────────────────────────────────────────────────────
tab_single, tab_fleet, tab_about = st.tabs([
    "Single Vehicle", "Fleet Analysis", "About",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: SINGLE VEHICLE
# ═══════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("Fill in the vehicle details below to get a full maintenance assessment.")

    vehicle_data = build_single_vehicle_input()

    if st.button("Analyze Vehicle", type="primary", use_container_width=True):
        with st.status("Running agent pipeline...", expanded=True) as status:
            st.write("Step 1/5 — ML prediction...")
            t0 = time.time()
            result = run_agent(vehicle_data, vehicle_id="User-Input")
            elapsed = time.time() - t0
            st.write("Steps 2-5 complete.")
            status.update(label=f"Analysis complete in {elapsed:.1f}s", state="complete")

        st.session_state["current_report"]       = result["final_report"]
        st.session_state["current_features"]     = vehicle_data
        st.session_state["current_agent_result"] = result
        st.session_state["chat_history"]         = []

    if "current_report" in st.session_state:
        st.markdown("---")
        render_report(
            st.session_state["current_report"],
            features=st.session_state.get("current_features"),
        )
        st.markdown("---")
        render_agent_trace(st.session_state["current_agent_result"])

        # Small prompt to use the sidebar chatbot
        st.info(
            "💬 Have questions about this report? Click the **Ask AI** button on "
            "the left to open the chat assistant."
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 2: FLEET ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_fleet:
    st.markdown(
        "Upload a CSV with multiple vehicles to get a fleet-wide analysis. "
        "Each row should match the columns below."
    )

    with st.expander("Required CSV format"):
        example_row = {
            "Brake_Condition": "0 (New) / 1 (Good) / 2 (Worn Out)",
            "Tire_Condition":  "0 (New) / 1 (Good) / 2 (Worn Out)",
            "Vehicle_Age":     "integer years",
            "Battery_Status":  "0 (New) / 1 (Good) / 2 (Weak)",
            "Reported_Issues": "integer 0-5",
            "Service_History": "integer 0-12",
            "Odometer_Reading":"integer km",
            "Insurance_Premium":"integer INR",
            "Accident_History":"integer 0-3",
            "Maintenance_History":"0 (Good) / 1 (Average) / 2 (Poor)",
        }
        st.table(pd.DataFrame([example_row]).T.rename(columns={0: "Values"}))

        sample = pd.DataFrame([
            {"Brake_Condition": 2, "Tire_Condition": 2, "Vehicle_Age": 9, "Battery_Status": 2,
             "Reported_Issues": 4, "Service_History": 2, "Odometer_Reading": 135000,
             "Insurance_Premium": 25000, "Accident_History": 2, "Maintenance_History": 2},
            {"Brake_Condition": 1, "Tire_Condition": 1, "Vehicle_Age": 3, "Battery_Status": 1,
             "Reported_Issues": 1, "Service_History": 4, "Odometer_Reading": 45000,
             "Insurance_Premium": 13000, "Accident_History": 0, "Maintenance_History": 0},
            {"Brake_Condition": 0, "Tire_Condition": 0, "Vehicle_Age": 1, "Battery_Status": 0,
             "Reported_Issues": 0, "Service_History": 2, "Odometer_Reading": 15000,
             "Insurance_Premium": 12000, "Accident_History": 0, "Maintenance_History": 0},
        ])
        buf = io.StringIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download sample CSV", buf.getvalue().encode(),
            file_name="fleet_sample.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload fleet CSV", type=["csv"])

    if uploaded is not None:
        try:
            fleet_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing = [c for c in FEATURE_ORDER if c not in fleet_df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
            st.stop()

        st.success(f"Loaded {len(fleet_df)} vehicles.")
        st.dataframe(fleet_df.head(10), use_container_width=True)

        max_rows = min(len(fleet_df), 20)
        n_to_analyze = st.slider(
            "How many vehicles to analyze?",
            1, max_rows, min(5, max_rows),
        )

        if st.button("Analyze Fleet", type="primary", use_container_width=True):
            results = []
            feature_rows = []
            progress = st.progress(0)
            status_txt = st.empty()

            for i in range(n_to_analyze):
                status_txt.text(f"Analyzing vehicle {i+1}/{n_to_analyze}...")
                row = fleet_df.iloc[i]
                vehicle_input = {k: row[k] for k in FEATURE_ORDER}
                vehicle_input = {k: int(v) if isinstance(v, (np.integer, int)) else v
                                 for k, v in vehicle_input.items()}
                feature_rows.append(vehicle_input)
                try:
                    res = run_agent(vehicle_input, vehicle_id=f"Vehicle-{i+1}")
                    results.append(res["final_report"])
                except Exception as e:
                    results.append({
                        "vehicle_id": f"Vehicle-{i+1}",
                        "risk_tier": "ERROR",
                        "executive_summary": f"Error: {e}",
                        "detailed_findings": [], "action_plan": [],
                        "risk_assessment": "", "preventive_recommendations": [],
                        "sources_cited": [], "probability": 0, "prediction": 0,
                        "triage_reasons": [], "disclaimer": "",
                        "vehicle_features": vehicle_input,
                    })
                progress.progress((i + 1) / n_to_analyze)

            status_txt.text("Fleet analysis complete.")

            st.markdown("---")
            st.markdown("## Fleet Overview")

            tier_counts = pd.Series(
                [r["risk_tier"] for r in results]
            ).value_counts().reindex(
                ["CRITICAL", "HIGH", "MEDIUM", "LOW", "ERROR"], fill_value=0,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Critical", int(tier_counts.get("CRITICAL", 0)))
            c2.metric("High",     int(tier_counts.get("HIGH", 0)))
            c3.metric("Medium",   int(tier_counts.get("MEDIUM", 0)))
            c4.metric("Low",      int(tier_counts.get("LOW", 0)))

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.bar(tier_counts.index, tier_counts.values,
                color=[TIER_COLORS.get(t, "#888") for t in tier_counts.index])
            ax.set_ylabel("Count")
            ax.set_title("Fleet Risk Distribution")
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.05, int(h),
                            ha="center", fontsize=11, color="white", fontweight="bold")
            st.pyplot(fig)

            st.markdown("## Per-Vehicle Reports")
            tier_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "ERROR": 4}
            sorted_pairs = sorted(
                zip(results, feature_rows),
                key=lambda pair: tier_rank.get(pair[0]["risk_tier"], 99),
            )
            for r, f in sorted_pairs:
                with st.expander(f"{r['vehicle_id']} — {r['risk_tier']}"):
                    render_report(r, features=f)

            st.markdown("## Export")
            export_df = pd.DataFrame([
                {
                    "Vehicle": r["vehicle_id"],
                    "Risk Tier": r["risk_tier"],
                    "ML Probability": f"{r.get('probability', 0):.2%}",
                    "Summary": r.get("executive_summary", "")[:200],
                    "Action Count": len(r.get("action_plan", [])),
                    "Sources": ", ".join(r.get("sources_cited", [])),
                }
                for r, _ in sorted_pairs
            ])
            st.dataframe(export_df, use_container_width=True)
            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button("Download fleet report as CSV",
                buf.getvalue().encode(), file_name="fleet_report.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ═══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## About this system

    This is an **agentic AI fleet management assistant** built on top of a
    classical ML model for vehicle maintenance prediction.

    ### Pipeline

    1. **Predict** — a trained Decision Tree (scikit-learn) scores the vehicle
    2. **Triage** — deterministic rules assign a risk tier (CRITICAL / HIGH / MEDIUM / LOW)
    3. **Retrieve** — relevant maintenance guidelines fetched via RAG (Chroma + sentence-transformers)
    4. **Reason** — Llama 3.3 70B (Groq) synthesizes a grounded detailed assessment
    5. **Report** — structured output with multiple sections, citations, and disclaimer

    A **sidebar chatbot** lets the user ask follow-up questions about the generated
    report. It has full report context and pulls fresh RAG excerpts for each question.

    ### Stack

    - **ML**: scikit-learn (Logistic Regression, Decision Tree)
    - **Agent**: LangGraph (stateful workflow)
    - **RAG**: Chroma vector DB + `sentence-transformers/all-MiniLM-L6-v2`
    - **LLM**: Groq (`llama-3.3-70b-versatile`)
    - **UI**: Streamlit

    ### Responsible AI

    - LLM output is constrained to JSON to minimize hallucination
    - All claims must cite a source document; retrieved docs are the only ground truth
    - Triage uses rule-based logic (not LLM) for safety-critical classification
    - Standard safety disclaimer attached to every report
    - Chatbot is scoped to report context — won't answer unrelated questions
    """)


# ── Always render the sidebar chatbot last ─────────────────────────
render_sidebar_chatbot()
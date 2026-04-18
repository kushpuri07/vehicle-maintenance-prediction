# Project 6: Vehicle Maintenance Prediction & Agentic Fleet Management

### From Condition-Based Diagnostics to Autonomous Fleet Operations

---

## Project Overview

This project implements an AI-driven fleet analytics system that predicts vehicle maintenance needs and extends into an autonomous fleet management assistant.

- **Milestone 1:** Classical machine learning applied to vehicle condition and usage data to predict maintenance need and identify the key drivers behind it.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about vehicle health, retrieves maintenance guidelines via RAG, generates structured intervention reports, and answers follow-up questions conversationally.

---

## Constraints & Requirements

- **Team Size:** 3 Students
- **API Budget:** Free Tier Only
- **Frameworks:** Scikit-learn (M1), LangGraph (M2)
- **Hosting:** Mandatory (Streamlit Cloud / Hugging Face Spaces)

---

## Technology Stack

| Component | Technology |
|---|---|
| **ML Models (M1)** | Logistic Regression, Decision Tree, Scikit-learn Pipelines |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Agent Framework (M2)** | LangGraph (stateful workflow) |
| **RAG (M2)** | Chroma vector DB + `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM (M2)** | Groq (`llama-3.3-70b-versatile`) |
| **UI Framework** | Streamlit |
| **Hosting** | Streamlit Cloud / Hugging Face Spaces |
| **Version Control** | GitHub |

---

## Architecture

### Milestone 1 Pipeline
```
Raw Dataset -> Cleaning -> Feature Encoding (Ordinal + Nominal)
           -> Stratified Train/Test Split -> Pipeline (Scaler + Model)
           -> Evaluation -> Persisted .pkl -> Streamlit UI
```

### Milestone 2 Agent Pipeline (LangGraph)
```
User Input (Form or CSV)
        |
        v
[Node 1: Predict]   - Loads trained model, outputs prediction + probability
        |
        v
[Node 2: Triage]    - Rule-based classification (CRITICAL/HIGH/MEDIUM/LOW)
        |
        v
[Node 3: Retrieve]  - Chroma RAG pulls relevant maintenance guidelines
        |
        v
[Node 4: Reason]    - Groq LLM synthesizes grounded assessment (JSON)
        |
        v
[Node 5: Report]    - Structured output with citations and disclaimer
        |
        v
Final Report + Interactive Chatbot (follow-up Q&A)
```

State flows between nodes as a `VehicleState` object, accumulating data at each step.

---

## Milestones & Deliverables

### Milestone 1: ML-Based Maintenance Prediction (Mid-Sem)

**Objective:** Predict whether a vehicle needs maintenance using classical ML pipelines on vehicle condition data — no LLMs involved.

**Key Deliverables:**
- Problem understanding and business context
- Dataset cleaning and engineering (internal consistency, realistic value ranges)
- Proper ordinal vs nominal encoding for categorical features
- Exploratory Data Analysis with visualizations
- Two trained models (Logistic Regression + Decision Tree) using Scikit-learn Pipelines
- Stratified train/test split with `class_weight='balanced'` for imbalance handling
- Model evaluation report (Accuracy, F1, Precision, Recall)
- Feature importance analysis
- Working deployed Streamlit application

**Results (on cleaned dataset):**

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Logistic Regression | 94.41% | 0.925 | Class-balanced, ordinal encoding |
| Decision Tree (depth=5) | 94.45% | 0.925 | Class-balanced, overfitting controlled |

Both models now show strong F1 on **both** classes — not just the majority — with precision/recall balanced across the 77/23 class distribution.

---

### Milestone 2: Agentic AI Fleet Management Assistant (End-Sem)

**Objective:** Extend the prediction system into an autonomous agentic assistant that reasons about vehicle health, retrieves maintenance best practices via RAG, and generates structured servicing recommendations.

**Key Deliverables:**
- Publicly deployed application (link in submission)
- LangGraph agent workflow (5 nodes with explicit state management)
- RAG system indexing 8 custom maintenance guideline documents
- Structured maintenance report with 8 sections:
  1. Executive Summary
  2. Vehicle Data Sheet
  3. Vehicle Health Overview (radar chart)
  4. Key Findings (with context)
  5. Recommended Actions (with rationale per action)
  6. Risk if Unaddressed
  7. Preventive Recommendations
  8. Sources Consulted
- Interactive follow-up chatbot (context-aware Q&A on generated reports)
- Fleet mode (CSV upload for batch analysis of multiple vehicles)
- GitHub Repository & complete codebase
- Demo Video (Max 5 mins)

**Responsible AI Measures:**
- LLM output constrained to JSON schema to minimize hallucination
- Every claim must cite a source document; retrieved excerpts are the only ground truth
- Triage tier uses rule-based logic (not LLM) for safety-critical classification
- Standard safety disclaimer attached to every report
- Chatbot scoped to report context — won't answer unrelated questions

---

## Dataset

- **Source:** Kaggle (cleaned for internal consistency)
- **Size:** 50,000 records × 20 columns
- **Target Variable:** `Need_Maintenance` (Binary: 0 or 1)
- **Class Distribution:** 77.2% need maintenance, 22.8% do not
- **Missing Values:** None

**Selected Features (with correlation to target in cleaned dataset):**

| Feature | Correlation |
|---|---|
| Brake_Condition | 0.75 |
| Tire_Condition | 0.72 |
| Vehicle_Age | 0.65 |
| Battery_Status | 0.56 |
| Reported_Issues | 0.55 |
| Service_History | 0.54 |
| Odometer_Reading | 0.51 |
| Insurance_Premium | 0.29 |
| Accident_History | 0.28 |
| Maintenance_History | 0.24 |

Dropped features (near-zero correlation, noise): Vehicle_Model, Fuel_Type, Transmission_Type, Owner_Type, Engine_Size, Mileage, Fuel_Efficiency.

---

## Project Structure

```
vehicle-maintenance-prediction/
├── app.py                              # Streamlit application (UI + chatbot)
├── model_dt.pkl                        # Trained Decision Tree pipeline
├── vehicle_dataset.csv                 # Cleaned dataset
├── vehicle_maintenance.ipynb           # ML training notebook
├── requirements.txt                    # Dependencies
├── README.md
├── .env.example                        # Template for API keys
├── .gitignore
│
├── agent/                              # LangGraph agent
│   ├── __init__.py
│   ├── state.py                        # VehicleState schema
│   ├── graph.py                        # Workflow definition
│   ├── chat.py                         # Follow-up chatbot
│   └── nodes/
│       ├── __init__.py
│       ├── predict.py                  # Node 1: ML prediction
│       ├── triage.py                   # Node 2: Rule-based tiering
│       ├── retrieve.py                 # Node 3: RAG retrieval
│       ├── reason.py                   # Node 4: LLM reasoning
│       └── report.py                   # Node 5: Report assembly
│
├── rag/                                # Retrieval system
│   ├── __init__.py
│   ├── build_index.py                  # Builds Chroma DB from docs
│   └── retriever.py                    # Query interface
│
├── maintenance_docs/                   # RAG knowledge base (8 docs)
│   ├── 01_brake_maintenance.md
│   ├── 02_tire_maintenance.md
│   ├── 03_battery_maintenance.md
│   ├── 04_reported_issues_guide.md
│   ├── 05_age_and_mileage.md
│   ├── 06_maintenance_and_accident_history.md
│   ├── 07_fleet_risk_tiers.md
│   └── 08_safety_disclaimers.md
│
└── chroma_db/                          # Vector DB (generated, gitignored)
```

---

## Setup & Running Locally

### 1. Clone and install
```bash
git clone <repo-url>
cd vehicle-maintenance-prediction
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Open .env and paste your Groq API key (get one at https://console.groq.com/keys)
```

### 3. Build the RAG vector database (one-time)
```bash
python rag/build_index.py
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Features

### Tab 1: Single Vehicle
Form-based input for a single vehicle. Outputs a detailed structured report with radar chart, plus a follow-up chatbot that can answer questions about the assessment.

### Tab 2: Fleet Analysis
Upload a CSV of multiple vehicles. Agent runs per row and produces a fleet-wide overview (risk distribution chart, per-vehicle reports sorted by priority, exportable CSV).

### Tab 3: About
Documentation on the pipeline architecture, stack, and responsible AI measures.

---

## Team

| Member | Milestone 1 Contributions | Milestone 2 Contributions |
|---|---|---|
| Vachan | Dataset sourcing, data preprocessing, label encoding, dropping irrelevant columns | Dataset cleaning for internal consistency, maintenance knowledge base authoring |
| Kush | EDA and visualizations, feature selection, model training with Pipelines, overfitting fix, model saving | LangGraph agent architecture, RAG setup with Chroma, prompt engineering |
| Aadit | Model evaluation, model comparison chart, Streamlit app development, GitHub setup and deployment | Extended Streamlit UI with tabs and chatbot, deployment, demo video |

---

## License

Academic project for educational purposes.
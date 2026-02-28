# Project 6: Vehicle Maintenance Prediction System

### From Condition-Based Diagnostics to Intelligent Maintenance Decisions

---

## Project Overview

This project involves the design and implementation of a machine learning system that predicts whether a vehicle requires maintenance based on its current condition indicators and historical data.

- **Milestone 1:** Classical machine learning techniques applied to real-world vehicle condition data to predict maintenance need and identify the key drivers behind it.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about vehicle health, retrieves maintenance best practices (RAG), and generates structured intervention recommendations.

---

## Constraints & Requirements

- **Team Size:** 3 Students
- **API Budget:** Free Tier Only
- **Framework:** Scikit-learn (M1) / LangGraph (M2)
- **Hosting:** Mandatory (Streamlit Cloud)

---

## Technology Stack

| Component | Technology |
|---|---|
| ML Models (M1) | Logistic Regression, Decision Tree, Scikit-learn Pipelines |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Model Persistence | Joblib |
| UI Framework | Streamlit |
| Hosting | Streamlit Cloud |
| Version Control | GitHub |
| Agent Framework (M2) | LangGraph, Chroma/FAISS (RAG) |
| LLMs (M2) | Open-source models or Free-tier APIs |

---

## Milestones & Deliverables

### Milestone 1: ML-Based Maintenance Prediction (Mid-Sem)

**Objective:** Predict whether a vehicle needs maintenance using classical ML pipelines on real vehicle condition data — no LLMs involved.

**Key Deliverables:**
- Problem understanding and business context
- Data preprocessing and feature selection pipeline
- Exploratory Data Analysis with visualizations
- Two trained models (Logistic Regression + Decision Tree) using Scikit-learn Pipelines
- Model evaluation report (Accuracy, F1, Precision, Recall)
- Working deployed application with UI (Streamlit)

**Results:**

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 88.77% | 0.9321 |
| Decision Tree | 96.38% | 0.9771 |

### Milestone 2: Agentic AI Maintenance Assistant (End-Sem)

**Objective:** Extend the system into an agentic assistant that reasons about vehicle health risk and retrieves best practices to generate structured maintenance recommendations.

**Key Deliverables:**
- Publicly deployed application (link required)
- Agent workflow documentation (States & Nodes)
- Structured maintenance report generation
- GitHub Repository & complete codebase
- Demo Video (Max 5 mins)

---

## Dataset

- **Source:** Kaggle
- **Size:** 50,000 records × 20 columns
- **Target Variable:** Need_Maintenance (Binary: 0 or 1)
- **Class Distribution:** 81% need maintenance, 19% do not
- **Missing Values:** None

**Selected Features:**

| Feature | Correlation with Target |
|---|---|
| Reported_Issues | 0.39 |
| Brake_Condition | 0.30 |
| Battery_Status | 0.29 |
| Service_History | — |
| Accident_History | — |
| Maintenance_History | — |

---

## Project Structure

```
vehicle-maintenance-prediction/
├── app.py                        # Streamlit application
├── model_dt.pkl                  # Saved Decision Tree pipeline
├── vehicle_dataset.csv           # Dataset
├── vehicle_maintenance.ipynb     # Main notebook
├── requirements.txt              # Dependencies
└── README.md
```

---

## Team

| Member | Contributions |
|---|---|
| Vachan | Dataset sourcing, data preprocessing, label encoding, dropping irrelevant columns |
| Kush | EDA and visualizations, feature selection, model training with Pipelines, overfitting fix, model saving |
| Aadit | Model evaluation, model comparison chart, Streamlit app development, GitHub setup and deployment |

---

## Live Demo

🔗 [Vehicle Maintenance Predictor — Streamlit App](https://your-app-link-here.streamlit.app)

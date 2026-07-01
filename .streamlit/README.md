# 🏦 Loan Approval Prediction System

An interactive, end-to-end Data Science web application built with **Python**, **Scikit-Learn**, and **Streamlit** that evaluates applicant risk profile metrics to predict financial loan eligibility.

---

## 📊 Project Overview
Evaluating loan applications manually is time-consuming and prone to human error. This system implements a predictive pipeline utilizing a **Random Forest Classifier** to assess risk profiles and instantly classify applications into **Approved** or **Rejected** states based on historical financial attributes.

### Key Capabilities:
* **Interactive Data Explorer**: Direct on-screen auditing of features, dynamic parameters, and statistical summaries.
* **Exploratory Data Analysis (EDA)**: Highly structural, automated charts plotting Income Distribution spreads, Loan-to-Income spreads, and baseline Approval Rates.
* **Model Evaluation Metrics**: Real-time evaluation reporting containing multi-class classification analytics, precision-recall breakdowns, and feature importance matrices.
* **Live Prediction Interface**: Interactive input layout enabling users to query custom application values for real-time model evaluation.

---

## ⚙️ Core Technical Architecture

### 🛠️ Tech Stack & Dependencies
* **Core Engine**: Python 3.10+
* **Data Pipelines**: Pandas, NumPy
* **Visual Representation**: Matplotlib, Seaborn
* **Predictive ML Modeling**: Scikit-Learn (Random Forest Engine)
* **Frontend UI Framework**: Streamlit Framework

### 🧩 Directory Layout
```text
├── .streamlit/
│   └── config.toml          # Custom modern UI styling configurations
├── loan_app.py              # Operational UI layout and pipeline routing logic
├── requirements.txt         # Production-level dependency environment definitions
└── README.md                # Comprehensive system documentation
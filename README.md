# ChiProtector: Real-Time Fraud & Anomaly Detection Engine

ChiProtector is a production-ready fraud and anomaly detection system designed to analyze financial transactions in real-time. Built on the SAML-D (Synthetic Anti-Money Laundering Dataset), it provides intelligent risk assessment through a high-performance REST API to identify suspicious activities and potential money laundering.
🚀 Key Features

    Real-Time Risk Scoring: Sub-millisecond inference using a FastAPI-powered backend.

    Behavioral Analytics: Computes complex entity statistics, including transaction frequency, sender Z-scores, and cross-border activity rates.

    Privacy-First Design: Protects sensitive data by using SHA-256 hashing for all account identifiers before they are processed or stored in artifacts.

    Three-Tier Decisioning: Provides clear, actionable decisions: Approve (Low Risk), Flag (Review Required), or Block (High Risk).

    Imbalance Handling: Optimized for fraud detection where fraudulent cases are rare, utilizing stratified sampling and scale_pos_weight.

🛠️ Technology Stack

    API Framework: FastAPI

    Machine Learning: LightGBM (Gradient Boosting), Scikit-Learn

    Data Processing: Pandas, NumPy

    Serialization: Joblib

📂 Project Structure

    app/features.py: Feature engineering pipeline and behavioral statistics logic.

    app/train.py: Model training script with automated threshold selection.

    app/api.py: FastAPI server for real-time transaction scoring.

    artifacts/: Directory for stored models, metrics, and behavioral baselines.

⚙️ Setup & Installation
1. Navigate to the Directory

First, ensure you are in the engine's root directory:
Bash

cd C:\path\fraud-anomaly-engine

2. Configure the Dataset

The engine requires the SAML-D.csv dataset for training. Set the path to your dataset file using an environment variable:

    Windows (PowerShell): $env:SAML_CSV_PATH="C:\path\to\SAML-D.csv"

    Linux/macOS: export SAML_CSV_PATH="/path/to/SAML-D.csv"
    
3. Install Dependencies

It is recommended to use a virtual environment:
Bash

python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
## OR 
.venv\Scripts\activate     # On Windows

pip install -r requirements.txt



📈 Usage
Step 1: Train the Model

Run the training script to build the behavioral baseline and the LightGBM classifier:
Bash

python -m app.train

This will generate model.joblib, sender_stats.joblib, and metrics.joblib inside the artifacts/ folder.
Step 2: Start the API Server

Launch the FastAPI server to begin processing transaction requests:
Bash

uvicorn app.api:app --reload

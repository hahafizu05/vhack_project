# Fraud & Anomaly Detection (SAML-D) – Model + Risk API

This repo trains a low-latency fraud/anomaly scorer on the SAML-D synthetic AML dataset and serves a **Risk API** that returns a **risk score** and a decision:

- **Approve**: low risk
- **Flag**: medium risk (manual review / step-up)
- **Block**: high risk

## Dataset

Expected CSV columns (as in SAML-D):

- `Time`, `Date`
- `Sender_account`, `Receiver_account`
- `Amount`
- `Payment_currency`, `Received_currency`
- `Sender_bank_location`, `Receiver_bank_location`
- `Payment_type`
- `Is_laundering` (label: 0/1)
- `Laundering_type` (optional for analysis; not used for training by default)

Put the dataset at:

- `C:\Users\Asus\Downloads\SAML-D.csv\SAML-D.csv`

or set `SAML_CSV_PATH` to point to it.

## Quickstart

Create a venv and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Train model (chunked reading; keeps all positives + samples negatives):

```bash
python -m app.train
```

Run API:

```bash
uvicorn app.api:app --reload
```

Score example:

```bash
curl -X POST http://127.0.0.1:8000/score ^
  -H "Content-Type: application/json" ^
  -d "{\"Time\":\"10:35:19\",\"Date\":\"2022-10-07\",\"Sender_account\":\"8724731955\",\"Receiver_account\":\"2769355426\",\"Amount\":1459.15,\"Payment_currency\":\"UK pounds\",\"Received_currency\":\"UK pounds\",\"Sender_bank_location\":\"UK\",\"Receiver_bank_location\":\"UK\",\"Payment_type\":\"Cash Deposit\"}"
```

## Privacy-first notes

- The API and artifacts store **hashed account IDs** (stable hashes) rather than raw IDs.
- No PII is logged by default.


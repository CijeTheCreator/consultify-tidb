# Consultify Medical Graph System

AI-powered medical consultation system using LangGraph workflows to connect patients with appropriate doctors and provide prescription assistance.

## Agent States

- **CLERKING**: Collecting patient symptoms, medical history, and determining specialist requirements
- **CONSULTING**: Doctor assigned, handling consultation and prescription assistance

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment variables:
- `TIDB_CONN_STRING`: TiDB database connection string
- `CLERK_SENDER_ID`: ID for the medical clerk agent

## Quick Start

1. Start the Flask API server:
```bash
python api.py
```

2. Test the health endpoint:
```bash
curl -X GET http://localhost:5500/health
```

3. Use the clerking endpoint for patient intake:
```bash
curl -X POST http://localhost:5500/clerking \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {
        "id": "msg_001",
        "original_content": "I have been experiencing chest pain for the last 2 days",
        "original_language": "en",
        "sender_id": "patient_123"
      }
    ],
    "consultation": {
      "id": "consultation_001",
      "state": "CLERKING",
      "patient_id": "patient_123"
    },
    "last_inserted_message_by_user": {
      "original_content": "I have been experiencing chest pain for the last 2 days",
      "original_language": "en",
      "sender_id": "patient_123"
    }
  }'
```

4. Use the prescription endpoint for drug recommendations:
```bash
curl -X POST http://localhost:5500/prescription \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {
        "id": "msg_001",
        "original_content": "I need medication for my hypertension",
        "original_language": "en",
        "sender_id": "patient_123"
      }
    ],
    "consultation": {
      "id": "consultation_001",
      "state": "CONSULTING",
      "patient_id": "patient_123"
    }
  }'
```

See `curls.md` for more detailed API testing examples and modify the commands for your specific use case.
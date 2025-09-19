# API Testing with cURL

This file contains cURL commands to test the clerking and prescription endpoints.

## Health Check

```bash
curl -X GET http://localhost:5500/health
```

## Clerking Endpoint

### Basic Clerking Request
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
      "id": "cmfp2xzur0003nra5fwtjrctv",
      "state": "CONSULTING",
      "patient_id": "cmfp2k0jo0002nra5ww0xkx1h"
    },
    "doctor": {
      "id": "doctor_001",
      "language": "fr",
      "specialty": "General Medicine"
    },
    "last_inserted_message_by_user": {
      "original_content": "I have been experiencing chest pain for the last 2 days",
      "original_language": "en",
      "sender_id": "cmfp2k0jo0002nra5ww0xkx1h"
    }
  }'
```

## Prescription Endpoint

### Basic Prescription Request
```bash
curl -X POST http://localhost:5500/prescription \
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
      "id": "cmfp2xzur0003nra5fwtjrctv",
      "state": "CONSULTING",
      "patient_id": "cmfp2k0jo0002nra5ww0xkx1h"
    },
    "doctor": {
      "id": "doctor_001",
      "language": "fr",
      "specialty": "General Medicine"
    },
    "last_inserted_message_by_user": {
      "original_content": "I have been experiencing chest pain for the last 2 days",
      "original_language": "en",
      "sender_id": "cmfp2k0jo0002nra5ww0xkx1h"
    }
  }'
```

## Notes

- Replace `http://localhost:5500` with your actual server URL if different
- Ensure the Flask app is running before executing these commands
- The API expects valid Pydantic model data according to the schemas in `models.py`
- All timestamps are automatically handled by the models if not provided
- **Both endpoints now run in background** - you receive immediate confirmation while the graph processes asynchronously
- Graph processing results/side effects will occur through the helper functions and database operations within the graphs

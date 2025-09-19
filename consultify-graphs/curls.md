# API Testing with cURL

This file contains cURL commands to test the clerking and prescription endpoints.

## Health Check

```bash
curl -X GET http://localhost:5000/health
```

## Clerking Endpoint

### Basic Clerking Request
```bash
curl -X POST http://localhost:5000/clerking \
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
      "id": "consult_001",
      "state": "CLERKING",
      "patient_id": "patient_123"
    },
    "next_message_to_append": {
      "id": "msg_002",
      "original_content": "Thank you for sharing that. Can you describe the chest pain in more detail?",
      "original_language": "en",
      "sender_id": "clerk_001"
    },
    "last_inserted_message_by_user": {
      "id": "msg_001",
      "original_content": "I have been experiencing chest pain for the last 2 days",
      "original_language": "en",
      "sender_id": "patient_123"
    }
  }'
```

### Advanced Clerking Request with Medical Specialty
```bash
curl -X POST http://localhost:5000/clerking \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {
        "id": "msg_001",
        "original_content": "I have severe chest pain and shortness of breath",
        "original_language": "en",
        "sender_id": "patient_123"
      },
      {
        "id": "msg_002",
        "original_content": "When did this start? Do you have any heart conditions?",
        "original_language": "en",
        "sender_id": "clerk_001"
      },
      {
        "id": "msg_003",
        "original_content": "It started yesterday evening. I have a history of high blood pressure",
        "original_language": "en",
        "sender_id": "patient_123"
      }
    ],
    "consultation": {
      "id": "consult_002",
      "state": "CLERKING",
      "patient_id": "patient_123"
    },
    "medical_specialty": "Cardiology",
    "next_message_to_append": {
      "id": "msg_004",
      "original_content": "Based on your symptoms, I will connect you with a cardiologist",
      "original_language": "en",
      "sender_id": "clerk_001"
    },
    "last_inserted_message_by_user": {
      "id": "msg_003",
      "original_content": "It started yesterday evening. I have a history of high blood pressure",
      "original_language": "en",
      "sender_id": "patient_123"
    }
  }'
```

## Prescription Endpoint

### Basic Prescription Request
```bash
curl -X POST http://localhost:5000/prescription \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {
        "id": "msg_001",
        "original_content": "I need medication for my high blood pressure",
        "original_language": "en",
        "sender_id": "patient_123"
      },
      {
        "id": "msg_002",
        "original_content": "What is your current blood pressure reading?",
        "original_language": "en",
        "sender_id": "doctor_001"
      },
      {
        "id": "msg_003",
        "original_content": "My last reading was 150/95",
        "original_language": "en",
        "sender_id": "patient_123"
      }
    ],
    "consultation": {
      "id": "consult_003",
      "state": "CONSULTING",
      "patient_id": "patient_123",
      "doctor_id": "doctor_001"
    }
  }'
```

### Advanced Prescription Request with Context
```bash
curl -X POST http://localhost:5000/prescription \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {
        "id": "msg_001",
        "original_content": "I have been diagnosed with Type 2 diabetes and need medication",
        "original_language": "en",
        "sender_id": "patient_456"
      },
      {
        "id": "msg_002",
        "original_content": "What is your HbA1c level and current glucose readings?",
        "original_language": "en",
        "sender_id": "doctor_002"
      },
      {
        "id": "msg_003",
        "original_content": "HbA1c is 8.2% and fasting glucose around 180 mg/dL",
        "original_language": "en",
        "sender_id": "patient_456"
      }
    ],
    "consultation": {
      "id": "consult_004",
      "state": "CONSULTING",
      "patient_id": "patient_456",
      "doctor_id": "doctor_002"
    },
    "query": "diabetes medication metformin dosage",
    "context_retrieved": "Metformin is first-line treatment for Type 2 diabetes. Starting dose is typically 500mg twice daily with meals."
  }'
```

## Error Testing

### Invalid JSON
```bash
curl -X POST http://localhost:5000/clerking \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}'
```

### Missing Required Fields
```bash
curl -X POST http://localhost:5000/prescription \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Response Examples

### Successful Response (Immediate Confirmation)
```json
{
  "success": true,
  "message": "Clerking graph started successfully",
  "consultation_id": "consult_001"
}
```

### Successful Prescription Response
```json
{
  "success": true,
  "message": "Prescription graph started successfully", 
  "consultation_id": "consult_003"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

## Notes

- Replace `http://localhost:5000` with your actual server URL if different
- Ensure the Flask app is running before executing these commands
- The API expects valid Pydantic model data according to the schemas in `models.py`
- All timestamps are automatically handled by the models if not provided
- **Both endpoints now run in background** - you receive immediate confirmation while the graph processes asynchronously
- Graph processing results/side effects will occur through the helper functions and database operations within the graphs
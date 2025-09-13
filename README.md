# Consultify

AI-powered medical consultation platform with multi-language support and prescription assistance.

## Architecture

- **Frontend**: Next.js 14 app with Prisma/TiDB integration
- **Backend**: LangGraph-based AI agents with Flask API
- **Database**: TiDB with vector search capabilities

## Workflows (@consultify-graphs/)

### Clerking Agent (`clerking.py`)
Medical consultation workflow that routes patients to appropriate doctors:

1. **Router**: Determines if enough patient info is gathered to assign doctor or needs more conversation
2. **Medical Query Generation**: Creates search queries for symptom/medical information
3. **Vector Retrieval**: Searches TiDB vector store (`microbiology_pharmacology_immunology_textbookV2`) 
4. **Document Grading**: Validates retrieval relevance, triggers query refinement if needed
5. **Specialty Determination**: Analyzes conversation to identify required medical specialty
6. **Doctor Matching**: Finds available doctors by specialty with fallback to general medicine
7. **Assignment**: Assigns doctor to consultation and generates rationale

### Prescription Agent (`prescriptionAgent.py`)
Drug recommendation workflow for active consultations:

1. **Query Generation**: Creates prescription-focused search queries
2. **Drug Retrieval**: Searches TiDB vector store (`british-formulary`)
3. **Document Grading**: Validates drug information relevance
4. **Prescription Generation**: Creates structured prescription recommendations
5. **Database Storage**: Saves prescriptions with timestamps and patient associations

### API Endpoints (`api.py`)
- `POST /clerking`: Triggers clerking workflow in background thread
- `POST /prescription`: Triggers prescription workflow in background thread  
- `GET /health`: Health check

## Data Flow

1. **Frontend** → Patient submits symptoms/queries
2. **API** → Receives consultation data, triggers appropriate LangGraph workflow
3. **LangGraph** → Processes through RAG pipeline using TiDB vector search
4. **Database** → Updates consultation state, messages, doctor assignments, prescriptions
5. **Frontend** → Polls for updates and displays results

## Setup

### Backend (consultify-graphs)
```bash
cd consultify-graphs
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set environment variables
export TIDB_CONN_STRING="your_tidb_connection"
export CLERK_SENDER_ID="clerk_user_id"
export MISTRAL_API_KEY="your_mistral_key"

python api.py
```

### Frontend (consultify-frontend)
```bash
cd consultify-frontend
npm install
npm run dev
```

## Environment Variables

### Backend
- `TIDB_CONN_STRING`: TiDB database connection string
- `CLERK_SENDER_ID`: ID for system clerk messages
- `MISTRAL_API_KEY`: Mistral AI API key

### Frontend
- Database and authentication configuration in `.env.local`

## Key Features

- **Multi-language Support**: Automatic translation based on doctor language preference
- **RAG-powered Responses**: Vector search through medical textbooks and drug formularies
- **State Management**: Real-time consultation state updates
- **Background Processing**: Non-blocking AI workflows
- **Fallback Logic**: Graceful handling when specific specialists unavailable
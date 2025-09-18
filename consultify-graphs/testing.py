from datetime import datetime
from clerking import *
from models import *

def create_sample_data():
    """Create sample data for testing the node"""
    
    # Sample consultation
    consultation = Consultation(
        id="consultation_123",
        state=ConsultationState.CLERKING,
        patient_id="patient_456",
        clerk_id="clerk_789",
        doctor_id="doctor_012",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Sample conversation messages
    conversation = [
        Message(
            id="msg_1",
            original_content="Hello, I've been having chest pain for the past 2 days",
            llm_content="Hello, I've been having chest pain for the past 2 days",
            sender_id="patient_456",
            consultation_id="consultation_123",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Message(
            id="msg_2", 
            original_content="Can you describe the chest pain? Is it sharp, dull, or crushing?",
            llm_content="Can you describe the chest pain? Is it sharp, dull, or crushing?",
            sender_id="clerk_789",
            consultation_id="consultation_123",
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Message(
            id="msg_3",
            original_content="It's a sharp pain that gets worse when I breathe deeply. It started after I was coughing a lot.",
            llm_content="It's a sharp pain that gets worse when I breathe deeply. It started after I was coughing a lot.",
            sender_id="patient_456", 
            consultation_id="consultation_123",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    
    # Last inserted message (most recent patient message)
    last_inserted_message = Message(
        id="msg_4",
        original_content="I'm also experiencing some shortness of breath and feel dizzy sometimes",
        llm_content="I'm also experiencing some shortness of breath and feel dizzy sometimes", 
        sender_id="patient_456",
        consultation_id="consultation_123",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Create the AgentState
    agent_state = AgentState(
        conversation=conversation,
        last_inserted_message_by_user=last_inserted_message,
        query="chest pain shortness of breath respiratory symptoms",
        refined_query="",
        context_retrieved="Chest pain can be caused by various conditions including pleuritic chest pain from respiratory infections, pneumonia, or pulmonary embolism. Sharp chest pain that worsens with deep breathing (pleuritic pain) combined with shortness of breath may indicate pleural inflammation or lung involvement. Respiratory infections can cause chest discomfort and breathing difficulties.",
        consultation=consultation,
        next_message_to_append=last_inserted_message
    )
    
    return agent_state

sample_data = create_sample_data()
response = refine_search_query(sample_data)
# print(response)

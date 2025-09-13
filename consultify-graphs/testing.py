from datetime import datetime
from clerking import *
from models import *
from dotenv import load_dotenv

def create_sample_data():
    """Create sample data for testing the node"""
    
    # Test values
    patient_id = "cmfp2k0jo0002nra5ww0xkx1h"
    doctor_id = "cmfp2k0jn0001nra56f4an5z0"
    consultation_id = "cmfp2xzur0003nra5fwtjrctv"
    clerk_id = os.getenv('CLERK_SENDER_ID', 'cmfp2fije0000nra5534qln9r')
    message_id = "cmfp36s6y0004nra5nvoajtjs"
    
    # Sample consultation
    consultation = Consultation(
        id=consultation_id,
        state=ConsultationState.CLERKING,
        patient_id=patient_id,
        clerk_id=clerk_id,
        doctor_id=doctor_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Sample conversation messages
    conversation = [
        Message(
            id="msg_1",
            original_content="Hello, I've been having chest pain for the past 2 days",
            llm_content="Hello, I've been having chest pain for the past 2 days",
            sender_id=patient_id,
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
            sender_id=patient_id, 
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
        sender_id=patient_id,
        consultation_id="consultation_123",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


    # next_message_to_append = Message(
    #     id=message_id,
    #     original_content="I'm also experiencing some shortness of breath and feel dizzy sometimes",
    #     llm_content="I'm also experiencing some shortness of breath and feel dizzy sometimes", 
    #     sender_id=patient_id,
    #     consultation_id=consultation_id,
    #     created_at=datetime.now(),
    #     updated_at=datetime.now()
    # )

    next_message_to_append = Message(
        original_content="I'm also experiencing some shortness of breath and feel dizzy sometimes",
        translated_language="en",
        llm_language="en",
        sender_id=patient_id,
        consultation_id=consultation_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    doctor = Doctor(
        id=doctor_id,
        language="fr",
        specialty="Pulmonology", 
    )

    
    # Create the AgentState
    agent_state = AgentState(
        conversation=conversation,
        last_inserted_message_by_user=last_inserted_message,
        query="chest pain shortness of breath respiratory symptoms",
        refined_query="",
        context_retrieved="Chest pain can be caused by various conditions including pleuritic chest pain from respiratory infections, pneumonia, or pulmonary embolism. Sharp chest pain that worsens with deep breathing (pleuritic pain) combined with shortness of breath may indicate pleural inflammation or lung involvement. Respiratory infections can cause chest discomfort and breathing difficulties.",
        consultation=consultation,
        next_message_to_append=next_message_to_append,
        medical_specialty="Cardiology",
        doctor=doctor,
        doctor_selection_rationale="A pulmonologist was selected because the patient's symptoms—sharp chest pain exacerbated by deep breathing and coughing—suggest a potential respiratory issue such as pleurisy or a pulmonary condition. The doctor's specialization in pulmonology ensures targeted expertise for diagnosing and treating lung-related concerns. While language preference (French) was noted, the primary alignment remains the doctor's relevant medical specialty."
    )
    
    return agent_state

sample_data = create_sample_data()
response = translate_message(sample_data)
# print(response)

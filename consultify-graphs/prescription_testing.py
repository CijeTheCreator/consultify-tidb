from datetime import datetime
from prescriptionAgent import *
from models import *
from dotenv import load_dotenv
import os
load_dotenv()

def create_sample_prescription_data():
    """Create sample data for testing the prescription agent"""
    
    # Test values
    patient_id = "cmfp2k0jo0002nra5ww0xkx1h"
    doctor_id = "cmfp2k0jn0001nra56f4an5z0"
    consultation_id = "cmfp2xzur0003nra5fwtjrctv"
    clerk_id = os.getenv('CLERK_SENDER_ID', 'cmfp2fije0000nra5534qln9r')
    
    # Sample consultation
    consultation = Consultation(
        id=consultation_id,
        state=ConsultationState.CONSULTING,
        prescription_assistance="active",
        prescription_assistance_state="active",
        patient_id=patient_id,
        clerk_id=clerk_id,
        doctor_id=doctor_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Sample conversation messages for prescription assistance
    conversation = [
        Message(
            id="msg_1",
            original_content="Doctor, I need help with prescribing medication for my patient's hypertension",
            llm_content="Doctor, I need help with prescribing medication for my patient's hypertension",
            sender_id=doctor_id,
            consultation_id=consultation_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Message(
            id="msg_2", 
            original_content="The patient is a 45-year-old male with newly diagnosed high blood pressure. No previous medications.",
            llm_content="The patient is a 45-year-old male with newly diagnosed high blood pressure. No previous medications.",
            sender_id=doctor_id,
            consultation_id=consultation_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Message(
            id="msg_3",
            original_content="What would be the most appropriate first-line antihypertensive medication and dosage?",
            llm_content="What would be the most appropriate first-line antihypertensive medication and dosage?",
            sender_id=doctor_id, 
            consultation_id=consultation_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    
    # Sample prescriptions that might be recommended
    sample_prescriptions = [
        Prescription(
            id="pres_1",
            drug_name="Amlodipine",
            frequency="Once daily",
            start_timestamp=datetime.now(),
            patient_id=patient_id,
            consultation_id=consultation_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Prescription(
            id="pres_2",
            drug_name="Lisinopril", 
            frequency="Once daily",
            start_timestamp=datetime.now(),
            patient_id=patient_id,
            consultation_id=consultation_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    
    # Create the PrescriptionAgentState
    prescription_state = PrescriptionAgentState(
        conversation=conversation,
        query="first-line antihypertensive medication dosage newly diagnosed hypertension",
        refined_query="",
        context_retrieved="For newly diagnosed hypertension in adults, first-line antihypertensive medications include ACE inhibitors (such as lisinopril 10mg once daily), calcium channel blockers (such as amlodipine 5mg once daily), thiazide diuretics, or ARBs. Consider patient age, ethnicity, and comorbidities when selecting initial therapy. Start with lowest effective dose and titrate based on response.",
        consultation=consultation,
        prescriptions_recommended=sample_prescriptions
    )
    
    return prescription_state

sample_state = create_sample_prescription_data()
reponse = generate_response(sample_state)

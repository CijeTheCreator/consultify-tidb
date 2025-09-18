import os
import requests
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_base_url() -> str:
    """Get the base URL from environment variables."""
    host = os.getenv('CONSULTIFY_HOST', 'http://localhost:3000')
    return f"{host}/api"

def create_message(
    sender_id: str,
    consultation_id: str,
    original_content: Optional[str] = None,
    original_language: Optional[str] = None,
    translated_content: Optional[str] = None,
    translated_language: Optional[str] = None,
    llm_content: Optional[str] = None,
    llm_language: Optional[str] = None,
    state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new message in a consultation.
    
    Args:
        sender_id: ID of the user sending the message
        consultation_id: ID of the consultation
        original_content: Original message content
        original_language: Language of original content
        translated_content: Translated message content
        translated_language: Language of translated content
        llm_content: LLM generated content
        llm_language: Language of LLM content
        state: Message state
    Returns:
        Dict containing the created message
    """
    url = f"{get_base_url()}/messages"
    payload = {
        "senderId": sender_id,
        "consultationId": consultation_id
    }
    
    # Add optional fields if provided
    if original_content is not None:
        payload["originalContent"] = original_content
    if original_language is not None:
        payload["originalLanguage"] = original_language
    if translated_content is not None:
        payload["translatedContent"] = translated_content
    if translated_language is not None:
        payload["translatedLanguage"] = translated_language
    if llm_content is not None:
        payload["llm_content"] = llm_content
    if llm_language is not None:
        payload["llm_language"] = llm_language
    if state is not None:
        payload["state"] = state
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def add_message(
    message_id: str,
    original_content: Optional[str] = None,
    original_language: Optional[str] = None,
    translated_content: Optional[str] = None,
    translated_language: Optional[str] = None,
    llm_content: Optional[str] = None,
    llm_language: Optional[str] = None,
    state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update an existing message (add content to it).
    
    Args:
        message_id: ID of the message to update
        original_content: Original message content
        original_language: Language of original content
        translated_content: Translated message content
        translated_language: Language of translated content
        llm_content: LLM generated content
        llm_language: Language of LLM content
        state: Message state
    
    Returns:
        Dict containing the updated message
    """
    url = f"{get_base_url()}/messages/{message_id}"

    payload = {}

    # Add fields if provided
    if original_content is not None:
        payload["originalContent"] = original_content
    if original_language is not None:
        payload["originalLanguage"] = original_language
    if translated_content is not None:
        payload["translatedContent"] = translated_content
    if translated_language is not None:
        payload["translatedLanguage"] = translated_language
    if llm_content is not None:
        payload["llm_content"] = llm_content
    if llm_language is not None:
        payload["llm_language"] = llm_language
    if state is not None:
        payload["state"] = state

    
    response = requests.patch(url, json=payload)
    response.raise_for_status()
    return response.json()

def update_message_state(message_id: str, state: str) -> Dict[str, Any]:
    """
    Update the state of a message.
    
    Args:
        message_id: ID of the message to update
        state: New state for the message
    
    Returns:
        Dict containing the updated message
    """
    url = f"{get_base_url()}/messages/{message_id}/state"
    
    payload = {"state": state}
    
    response = requests.patch(url, json=payload)
    response.raise_for_status()
    return response.json()

def assign_doctor(consultation_id: str, doctor_id: str) -> Dict[str, Any]:
    """
    Assign a doctor to a consultation.
    
    Args:
        consultation_id: ID of the consultation
        doctor_id: ID of the doctor to assign
    
    Returns:
        Dict containing the updated consultation
    """
    url = f"{get_base_url()}/consultations/{consultation_id}/assign-doctor"
    
    payload = {"doctorId": doctor_id}
    
    response = requests.patch(url, json=payload)
    response.raise_for_status()
    return response.json()

def get_doctors_by_specialty(specialty: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get doctors, optionally filtered by specialty.
    
    Args:
        specialty: Optional specialty to filter by
    
    Returns:
        List of doctor dictionaries
    """
    url = f"{get_base_url()}/doctors"
    params = {}
    
    if specialty is not None:
        params["specialty"] = specialty
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def get_user(user_id: str) -> Dict[str, Any]:
    """
    Get a user by ID.
    
    Args:
        user_id: ID of the user to retrieve
    
    Returns:
        Dict containing user information
    """
    url = f"{get_base_url()}/users"
    params = {"id": user_id}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def create_user(
    user_type: str,
    language: str,
    specialty: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new user.
    
    Args:
        user_type: Type of user ('DOCTOR' or 'PATIENT')
        language: User's language
        specialty: Specialty (required for doctors)
    
    Returns:
        Dict containing the created user
    """
    url = f"{get_base_url()}/users"
    
    payload = {
        "type": user_type,
        "language": language
    }
    
    if specialty is not None:
        payload["specialty"] = specialty
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

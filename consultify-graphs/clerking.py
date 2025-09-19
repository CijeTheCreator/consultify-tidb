import os
from langgraph.graph import StateGraph, START, END, message
from langgraph.prebuilt import ToolNode
from models import Message
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, cast
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import TiDBVectorStore
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from models import *
from prompts import *
from helpers import add_message, get_doctors_by_specialty, create_message, assign_doctor, format_conversation_history
import random
load_dotenv()

# Initialize the same embedding model you used before
embeddings = MistralAIEmbeddings()
response_model = ChatMistralAI(model = "mistral-large-latest")

# Connect to existing vector store
tidb_connection_string = os.getenv("TIDB_CONN_STRING") or ""
vector_store = TiDBVectorStore(
    connection_string=tidb_connection_string,
    embedding_function=embeddings,
    table_name="microbiology_pharmacology_immunology_textbookV2",  # Your existing table name
    distance_strategy="cosine"
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Number of documents to retrieve
)

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_info_on_symptoms",
    "Search on return information about symptoms",
)


def create_medical_lookup_query_or_respond(state: AgentState):
    """Generate a query and then use it to retrieve medical information."""
    
    # Create message if it doesn't exist
    if not state.next_message_to_append or not state.next_message_to_append.id:
        clerk_sender_id = os.getenv("CLERK_SENDER_ID")
        if clerk_sender_id and state.consultation:
            routing_message = create_message(
                sender_id=clerk_sender_id,
                consultation_id=state.consultation.id,
                original_content="",
                original_language="",
                translated_content="",
                translated_language="",
                llm_content="",
                llm_language="en",
                state="Generating medical query"
            )
            print(f"New message is {routing_message}")
            # Update state with the new message
            next_message_to_append = Message(
                id=routing_message.get('id'),
                sender_id=clerk_sender_id,
                consultation_id=state.consultation.id,
                original_content=routing_message.get('original_content', ""),
                original_language=routing_message.get('original_language', ""),
                translated_content=routing_message.get('translated_content', ""),
                translated_language=routing_message.get('translated_language', ""),
                llm_content=routing_message.get('llm_content', ""),
                llm_language=routing_message.get('llm_language', "en")
            )
    else:
        # Update message state to show we're generating query
        print("Generating Medical Query")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Generating medical query"
        )
    
    # Step 1: Check if refined_query is available, otherwise generate a new query
    if state.refined_query:
        # Use the refined query from the rewrite step
        query_to_use = state.refined_query
    else:
        # Generate a new query using the query generation prompt
        last_message = state.last_inserted_message_by_user
        conversation_history = format_conversation_history(state.conversation, state.consultation)
        
        last_message_content = last_message.original_content or last_message.llm_content
        query_prompt = GENERATE_QUERY_PROMPT.replace("<LastQuestion>", last_message_content).replace("</LastQuestion>", "").replace("<FullConversation>", conversation_history).replace("</FullConversation>", "")
        
        query_response = response_model.with_structured_output(QueryGeneration).invoke([
            {"role": "user", "content": query_prompt}
        ])
        query_to_use = cast(QueryGeneration, query_response).query
    
    # Update message state to show we're retrieving information
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Retrieving medical information")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Retrieving medical information"
        )
    
    # Step 2: Use the query with the retriever tool
    retrieval_response = retriever_tool.invoke(query_to_use)
    
    # Update message state to show retrieval completed
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Information retrieved")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Information retrieved"
        )
    
    # Step 3: Update state with the query and retrieved context
    return {
        "query": query_to_use,
        "context_retrieved": retrieval_response,
        "conversation": state.conversation + [state.last_inserted_message_by_user],
        "next_message_to_append": next_message_to_append
    }


def grade_documents(
    state: AgentState,
) -> Literal["generate_response", "refine_search_query"]:
    """Determine whether the retrieved documents are relevant to the question."""
    
    # Update message state to show we're grading documents
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Evaluating retrieved information")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Evaluating retrieved information"
        )
    
    prompt = GRADE_PROMPT.format(
        query=state.query,
        context_retrieved=state.context_retrieved
    )
    print(f"Prompt: {prompt}")
    grade_response = response_model.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": prompt}
    ])
    
    grade = cast(GradeDocuments, grade_response).binary_score
    
    # Update message state based on grade result
    if state.next_message_to_append and state.next_message_to_append.id:
        if grade == "YES":
            print("Information verified - generating response")
            add_message(
                message_id=state.next_message_to_append.id,
                state="Information verified - generating response"
            )
        else:
            print("Refining search query")
            add_message(
                message_id=state.next_message_to_append.id,
                state="Refining search query"
            )
    
    if grade == "YES":
        return "generate_response"
    else:
        return "refine_search_query"


class DoctorSelection(BaseModel):
  """Determine if it is time to select a doctor based on the current state"""
  binary_answer: str = Field(
      description="Has gathered enough data to pick a doctor: 'yes' if there is enough data, or 'no' if there isn't enough data")

def router(state: AgentState)-> Literal["create_medical_lookup_query_or_respond", "determine_required_medical_specialty", "translate_message"]:
  """Determine if it is time to select a doctor, respond to a user or translate a message based on the current state"""
  
  # If consultation is already in CONSULTING state, translate the message
  if state.consultation.state == "CONSULTING": 
        return "translate_message"
  
  # Format conversation history for analysis
  conversation = format_conversation_history(state.conversation, state.consultation)
  
  # Use the doctor selection prompt to determine if enough info has been gathered
  prompt = DOCTOR_SELECTION_PROMPT.format(conversation=conversation)
  response = (
      response_model
      .with_structured_output(DoctorSelection).invoke(
          [{"role": "user", "content": prompt}]
      ))
  
  # Check the binary answer from the LLM
  score = response.binary_answer
  
  # Route based on whether enough information has been gathered
  if score == "yes":
    # Enough information gathered - proceed to doctor selection
    return "determine_required_medical_specialty"
  else:
    # Need more information - continue conversation with medical lookup/response
    return "create_medical_lookup_query_or_respond"


class RewrittenQuestion(BaseModel):
    """Pydantic model for the rewritten question response."""
    improved_question: str = Field(
        description="The semantically improved and clarified version of the original question"
    )


def refine_search_query(state: AgentState):
    """Rewrite the original user question using structured output."""
    
    # Update message state to show we're refining the query
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Refining search query")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Refining search query"
        )
    
    # Use the REWRITER_PROMPT which expects previous query and context
    prompt = REWRITER_PROMPT.replace("<PreviousQuery>", state.query or "").replace("</PreviousQuery>", "").replace("<PreviousContextRetrieved>", state.context_retrieved or "").replace("</PreviousContextRetrieved>", "")
    
    # Configure the model to use structured output with Pydantic
    response = response_model.with_structured_output(RewrittenQuestion).invoke([
        {"role": "user", "content": prompt}
    ])
    improved_question = cast(RewrittenQuestion, response).improved_question
    
    # Update message state to show query has been refined
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Search query refined")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Search query refined"
        )
    
    # Store the improved question in refined_query field
    print(f"Refined Query: {improved_question}")
    return {"refined_query": improved_question}

def generate_medical_consulatation_summary(state: AgentState):
  """Generate a medical consultation summary"""
  return {"messages": [{"role": "user", "content": "Medical consultation summary generated"}]}

def determine_required_medical_specialty(state: AgentState):
    """Determine the required medical specialty"""
    
    # Create message if it doesn't exist
    if not state.next_message_to_append or not state.next_message_to_append.id:
        clerk_sender_id = os.getenv("CLERK_SENDER_ID")
        if clerk_sender_id and state.consultation:
            routing_message = create_message(
                sender_id=clerk_sender_id,
                consultation_id=state.consultation.id,
                original_content="",
                original_language="",
                translated_content="",
                translated_language="",
                llm_content="",
                llm_language="en",
                state="Determining required medical specialty"
            )
            # Update state with the new message
            from models import Message
            state.next_message_to_append = Message(
                id=routing_message.get('id'),
                sender_id=clerk_sender_id,
                consultation_id=state.consultation.id,
                original_content="",
                original_language="",
                translated_content="",
                translated_language="",
                llm_content="",
                llm_language="en"
            )
    else:
        # Update message state to show we're determining specialty
        print("Determining required medical specialty")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Determining required medical specialty"
        )
    
    # Format conversation history
    conversation_history = format_conversation_history(state.conversation, state.consultation)
    
    # Prepare the specialist selection prompt
    prompt = SELECT_SPECIALIST_PROMPT.replace(
        "<ConversationSoFar>", conversation_history
    ).replace(
        "{Insert conversation here}", ""
    ).replace(
        "</ConversationSoFar>", ""
    )
    
    # Call LLM with structured output to get the specialty
    specialty_response = response_model.with_structured_output(MedicalSpecialty).invoke([
        {"role": "user", "content": prompt}
    ])
    
    selected_specialty = cast(MedicalSpecialty, specialty_response).specialty
    print(f"Specialty selected is {selected_specialty}")
    
    # Update message state to show specialty determined
    if state.next_message_to_append and state.next_message_to_append.id:
        print(f"Specialty determined: {selected_specialty}")
        add_message(
            message_id=state.next_message_to_append.id,
            state=f"Specialty determined: {selected_specialty}"
        )
    
    return {"medical_specialty": selected_specialty}

def find_matching_doctor(state: AgentState):
    """Find a matching doctor based on medical specialty with fallback strategy"""
    specialty = state.medical_specialty
    
    # Update message state to show we're finding a doctor
    if state.next_message_to_append and state.next_message_to_append.id:
        print(f"Finding doctor with specialty: {specialty}")
        add_message(
            message_id=state.next_message_to_append.id,
            state=f"Finding doctor with specialty: {specialty}"
        )
    
    # First, try to find doctors with the specific specialty
    doctors = get_doctors_by_specialty(specialty)
    
    # If no doctors found with specific specialty, try "General Medicine"
    if not doctors:
        print(f"No doctors found for specialty '{specialty}', trying 'General Medicine'")
        if state.next_message_to_append and state.next_message_to_append.id:
            print(f"Finding general medicine doctor")
            add_message(
                message_id=state.next_message_to_append.id,
                state="Finding general medicine doctor"
            )
        doctors = get_doctors_by_specialty("General Medicine")
    
    # If still no doctors found, get all doctors
    if not doctors:
        print("No doctors found for 'General Medicine', getting all doctors")
        if state.next_message_to_append and state.next_message_to_append.id:
            print(f"Finding any available doctor")
            add_message(
                message_id=state.next_message_to_append.id,
                state="Finding any available doctor"
            )
        doctors = get_doctors_by_specialty()
    
    # If we still have no doctors, raise an error
    if not doctors:
        raise ValueError("No doctors available in the system")
    
    # Select a random doctor from the available list
    selected_doctor_data = random.choice(doctors)
    
    # Create Doctor model instance from API response
    selected_doctor = Doctor(
        id=selected_doctor_data.get('id'),
        language=selected_doctor_data.get('language'),
        specialty=selected_doctor_data.get('specialty')
    )
    
    print(f"Selected doctor: {selected_doctor.id} with specialty: {selected_doctor.specialty}")
    
    # Update message state to show doctor found
    if state.next_message_to_append and state.next_message_to_append.id:
        print(f"Doctor found: {selected_doctor.specialty} specialist")
        add_message(
            message_id=state.next_message_to_append.id,
            state=f"Doctor found: {selected_doctor.specialty} specialist"
        )
    
    return {"doctor": selected_doctor}

def create_doctor_selection_rationale(state: AgentState):
    """Create a doctor selection rationale"""
    
    # Update message state to show we're creating rationale
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Creating doctor selection rationale")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Creating doctor selection rationale"
        )
    
    # Format conversation history
    conversation_history = format_conversation_history(state.conversation, state.consultation)
    
    # Format doctor information
    doctor_info = f"Doctor ID: {state.doctor.id}, Specialty: {state.doctor.specialty}, Language: {state.doctor.language}"
    
    # Prepare the prompt with conversation and doctor details
    prompt = SELECTION_RATIONALE_PROMPT.replace(
        "<ConversationWithClerk>", conversation_history
    ).replace(
        "[Insert conversation between patient and medical clerk here]", ""
    ).replace(
        "</ConversationWithClerk>", ""
    ).replace(
        "<DoctorSelected>", doctor_info
    ).replace(
        "[Insert doctor's profile, specialization, experience, and relevant details here]", ""
    ).replace(
        "</DoctorSelected>", ""
    )
    
    # Generate rationale using structured output
    rationale_response = response_model.with_structured_output(DoctorSelectionRationale).invoke([
        {"role": "user", "content": prompt}
    ])
    
    generated_rationale = cast(DoctorSelectionRationale, rationale_response).rationale
    print(f"Doctor Selection Rationale: {generated_rationale}")
    
    # Update message state to show rationale created
    if state.next_message_to_append and state.next_message_to_append.id:
        add_message(
            message_id=state.next_message_to_append.id,
            state="Doctor selection rationale created"
        )
    
    return {"doctor_selection_rationale": generated_rationale}

def assign_doctor_to_consultation(state: AgentState):
    """Assign the doctor to the consultation"""
    
    # Update message state to show we're assigning doctor
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Assigning doctor to consultation")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Assigning doctor to consultation"
        )
    
    # Get sender ID from environment variable
    clerk_sender_id = os.getenv("CLERK_SENDER_ID")
    if not clerk_sender_id:
        raise ValueError("CLERK_SENDER_ID environment variable is not set")
    
    # Get the rationale text from state
    rationale_text = state.doctor_selection_rationale
    if not rationale_text:
        raise ValueError("No doctor_selection_rationale found in state")


    message_id = state.next_message_to_append.id
    if not message_id:
        raise ValueError("No message id for next message")
    
    # Determine original language
    original_language = (
        state.next_message_to_append.original_language 
        if state.next_message_to_append and state.next_message_to_append.original_language 
        else "en"
    )
    
    # Create message using create_message function
    add_message(
        message_id=message_id,
        sender_id=clerk_sender_id,
        consultation_id=state.consultation.id,
        original_content=rationale_text,
        original_language=original_language,
        translated_content=rationale_text,
        translated_language=original_language,
        llm_content=rationale_text,
        llm_language=original_language
    )
    
    # Assign the doctor to the consultation
    assign_doctor(
        consultation_id=state.consultation.id,
        doctor_id=state.doctor.id
    )
    print("Doctor Assigned")
    
    # Update message state to show doctor assigned
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Doctor successfully assigned")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Doctor successfully assigned"
        )

def generate_response(state: AgentState):
    """Generate an answer."""
    
    # Update message state to show we're generating response
    if state.next_message_to_append and state.next_message_to_append.id:
        print("Generating response")
        add_message(
            message_id=state.next_message_to_append.id,
            state="Generating response"
        )
    
    # Format conversation history
    conversation_history = format_conversation_history(state.conversation, state.consultation)
    
    # Prepare the prompt with conversation and research findings
    prompt = GENERATE_MESSAGE_PROMPT.replace(
        "<ConversationSoFar>", conversation_history
    ).replace(
        "</ConversationSoFar>", ""
    ).replace(
        "<FindingsFromResearch>", state.context_retrieved or ""
    ).replace(
        "</FindingsFromResearch>", ""
    )
    
    # Generate message using structured output
    message_response = response_model.with_structured_output(GeneratedMessage).invoke([
        {"role": "user", "content": prompt}
    ])
    generated_content = cast(GeneratedMessage, message_response).message_content
    print("Reponse Generated")
    print(f"Generated Response: {generated_content}")
    
    # Get message ID from the state
    if not state.next_message_to_append or not state.next_message_to_append.id:
        raise ValueError("No message ID found in next_message_to_append")
    
    message_id = state.next_message_to_append.id
    
    # Get original language from the last user message
    original_language = (
        state.next_message_to_append.original_language 
        if state.next_message_to_append and state.next_message_to_append.original_language 
        else "en"
)
    
    print(f"Translated Language: {original_language}")
    # Update the existing message with all required content
    updated_message = add_message(
        message_id=message_id,
        original_content=generated_content,
        original_language=original_language,
        translated_content=generated_content,
        translated_language=original_language,
        llm_content=generated_content,
        llm_language=original_language,
        state="Response generated"
    )
    print("Message sent")
    
    # Update the message object in state with the generated content
    updated_message_obj = state.next_message_to_append
    updated_message_obj.original_content = generated_content
    updated_message_obj.original_language = original_language
    updated_message_obj.translated_content = generated_content
    updated_message_obj.translated_language = original_language
    updated_message_obj.llm_content = generated_content
    updated_message_obj.llm_language = original_language
    
    return {"next_message_to_append": updated_message_obj}

workflow = StateGraph(AgentState)

def translate_message(state: AgentState):
    """Translate message to target language and llm language"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Get the message to translate from last_inserted_message_by_user
    if not state.last_inserted_message_by_user or not state.last_inserted_message_by_user.original_content:
        raise ValueError("No message to translate found in last_inserted_message_by_user")
    
    original_content = state.last_inserted_message_by_user.original_content
    sender_id = state.last_inserted_message_by_user.sender_id
    
    # Create message first with "Translating" state
    created_message = create_message(
        sender_id=sender_id,
        consultation_id=state.consultation.id,
        original_content=original_content,
        original_language=state.last_inserted_message_by_user.original_language or "en",
        state="Translating"
    )
    
    # Create new Message object for next_message_to_append
    from models import Message
    next_message_to_append = Message(
        id=created_message.get('id'),
        sender_id=sender_id,
        consultation_id=state.consultation.id,
        original_content=original_content,
        original_language=state.last_inserted_message_by_user.original_language or "en",
        translated_content="",
        translated_language="",
        llm_content="",
        llm_language=""
    )
    
    print("Translating")
    
    # Determine target languages
    target_language = "en"  # Default to English
    llm_language = "en"     # LLM always processes in English
    
    # If doctor is assigned, use doctor's language as target language
    if hasattr(state, 'doctor') and state.doctor and state.doctor.language:
        target_language = state.doctor.language
    
    # Update message state to show language determination
    print(f"Translating to {target_language}")
    add_message(
        message_id=next_message_to_append.id,
        state=f"Translating to {target_language}"
    )
    
    def translate_to_language(content: str, language: str) -> str:
        """Translate content to specified language"""
        if language == next_message_to_append.original_language:
            return content  # No translation needed if same language
            
        prompt = TRANLSATION_PROMPT.replace("[TARGET_LANGUAGE]", language).replace("[MESSAGE_CONTENT]", content)
        
        translation_response = response_model.with_structured_output(TranslatedMessage).invoke([
            {"role": "user", "content": prompt}
        ])
        return cast(TranslatedMessage, translation_response).translated_content
    
    # Perform translations in parallel if different languages
    if target_language == llm_language:
        # Same language, only translate once
        translated_content = translate_to_language(original_content, target_language)
        llm_content = translated_content
    else:
        # Different languages, translate in parallel
        # Update message state to show parallel translation
        print("Performing parallel translations")
        add_message(
            message_id=next_message_to_append.id,
            state="Performing parallel translations"
        )
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            target_future = executor.submit(translate_to_language, original_content, target_language)
            llm_future = executor.submit(translate_to_language, original_content, llm_language)
            
            translated_content = target_future.result()
            llm_content = llm_future.result()
    
    # Update the message object
    next_message_to_append.translated_content = translated_content
    next_message_to_append.translated_language = target_language
    next_message_to_append.llm_content = llm_content
    next_message_to_append.llm_language = llm_language
    
    # Add translations to the original message using add_message
    print("Translated")
    print(f"Translated Message: {translated_content}")
    add_message(
        message_id=next_message_to_append.id,
        translated_content=translated_content,
        translated_language=target_language,
        llm_content=llm_content,
        llm_language=llm_language,
        state="Translated"
    )
    
    return {"next_message_to_append": next_message_to_append}

# Define the nodes we will cycle between
workflow.add_node(translate_message)
workflow.add_node(assign_doctor_to_consultation)
workflow.add_node(create_doctor_selection_rationale)
workflow.add_node(determine_required_medical_specialty)
workflow.add_node(find_matching_doctor)
workflow.add_node(generate_medical_consulatation_summary)
workflow.add_node(create_medical_lookup_query_or_respond)
workflow.add_node("search_mpi_textbook", ToolNode([retriever_tool]))
workflow.add_node(refine_search_query)
workflow.add_node(generate_response)
workflow.add_conditional_edges(
    START,
    router
)
workflow.add_edge("determine_required_medical_specialty", "find_matching_doctor")
workflow.add_edge("find_matching_doctor", "create_doctor_selection_rationale")
workflow.add_edge("create_doctor_selection_rationale", "generate_medical_consulatation_summary")
workflow.add_edge("generate_medical_consulatation_summary", "assign_doctor_to_consultation")
# Connect directly to generate_response since we call the tool within the function
workflow.add_edge("create_medical_lookup_query_or_respond", "generate_response")
# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "search_mpi_textbook",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_response", END)
workflow.add_edge("determine_required_medical_specialty", END)
workflow.add_edge("refine_search_query", "create_medical_lookup_query_or_respond")

# Compile
graph = workflow.compile()


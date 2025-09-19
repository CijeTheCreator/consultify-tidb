import os
from typing import cast
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import TiDBVectorStore
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from models import *
from prompts import *
from helpers import format_conversation_history, update_prescription_assistance_state, add_prescription_assistance, create_prescription
load_dotenv()


# Initialize the same embedding model you used before
embeddings = MistralAIEmbeddings()
response_model = ChatMistralAI(model = "mistral-large-latest")

# Connect to existing vector store
tidb_connection_string = os.getenv("TIDB_CONN_STRING") or ""
vector_store = TiDBVectorStore(
    connection_string=tidb_connection_string,
    embedding_function=embeddings,
    table_name="british-formulary",  # Your existing table name
    distance_strategy="cosine"
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Number of documents to retrieve
)

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_info_on_drugs",
    "Search and return information about drugs",
)

response_model = ChatMistralAI(model = "mistral-large-latest")

def create_prescription_lookup_query(state: PrescriptionAgentState):
    """Generate a query and then use it to retrieve prescription information."""
    
    # Update state to inform user about current node activity
    if state.consultation:
        update_prescription_assistance_state(state.consultation.id, "Generating search query for drug information")
    
    # Step 1: Check if refined_query is available, otherwise generate a new query
    if state.refined_query:
        # Use the refined query from the rewrite step
        query_to_use = state.refined_query
    else:
        # Generate a new query using the prescription query generation prompt
        if not state.conversation or len(state.conversation) == 0:
            raise ValueError("No conversation available to generate query")
        
        # Get the last user message
        last_message = state.conversation[-1] if state.conversation else None
        if not last_message:
            raise ValueError("No last message found in conversation")
        
        # Format conversation history using helper function
        if state.consultation:
            conversation_history = format_conversation_history(state.conversation, state.consultation)
        else:
            # Fallback to simplified format if no consultation context available
            conversation_history = "\n".join([
                f"Message: {msg.original_content or msg.llm_content or ''}" 
                for msg in state.conversation
            ])
        
        last_message_content = last_message.original_content or last_message.llm_content or ""
        query_prompt = PRESCRIPTION_QUERY_PROMPT.replace(
            "<UserQuestion>", last_message_content
        ).replace(
            "</UserQuestion>", ""
        ).replace(
            "<ConversationHistory>", conversation_history
        ).replace(
            "</ConversationHistory>", ""
        )
        
        query_response = response_model.with_structured_output(QueryGeneration).invoke([
            {"role": "user", "content": query_prompt}
        ])
        query_to_use = query_response.query
        print(f"Query Generated")
    
    # Step 2: Use the query with the retriever tool
    retrieval_response = retriever_tool.invoke(query_to_use)
    print(f"Retrieval")
    
    # Step 3: Update state with the query and retrieved context
    return {
        "query": query_to_use,
        "context_retrieved": retrieval_response
    }


def grade_documents(
    state: PrescriptionAgentState,
) -> Literal["generate_response", "refine_search_query"]:
    """Determine whether the retrieved documents are relevant to the question."""
    
    prompt = GRADE_PROMPT.format(
        query=state.query,
        context_retrieved=state.context_retrieved
    )
    grade_response = response_model.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": prompt}
    ])
    
    grade = cast(GradeDocuments, grade_response).binary_score
    if grade == "YES":
        return "generate_response"
    else:
        return "refine_search_query"


def refine_search_query(state: PrescriptionAgentState):
    """Rewrite the original user question using structured output."""
    
    # Update state to inform user about current node activity
    if state.consultation:
        update_prescription_assistance_state(state.consultation.id, "Refining search query for better drug information retrieval")
    
    # Use the REWRITER_PROMPT which expects previous query and context
    prompt = REWRITER_PROMPT.replace("<PreviousQuery>", state.query or "").replace("</PreviousQuery>", "").replace("<PreviousContextRetrieved>", state.context_retrieved or "").replace("</PreviousContextRetrieved>", "")
    
    # Configure the model to use structured output with Pydantic
    response = response_model.with_structured_output(RewrittenQuestion).invoke([
        {"role": "user", "content": prompt}
    ])
    improved_question = cast(RewrittenQuestion, response).improved_question
    
    # Store the improved question in refined_query field
    print(f"Refined Query")
    return {"refined_query": improved_question}


def generate_response(state: PrescriptionAgentState):
    """Generate the prescription recommendation"""
    
    # Update state to inform user about current node activity
    if state.consultation:
        update_prescription_assistance_state(state.consultation.id, "Generating prescription recommendations")
    
    # Format conversation history using helper function
    if state.consultation:
        conversation_history = format_conversation_history(state.conversation, state.consultation)
    else:
        # Fallback to simplified format if no consultation context available
        conversation_history = "\n".join([
            f"Message: {msg.original_content or msg.llm_content or ''}" 
            for msg in state.conversation or []
        ])
    
    # Prepare the prompt with conversation history and retrieved context
    prompt = PRESCRIPTION_RECOMMENDATION_PROMPT.format(
        conversation_history=conversation_history,
        context_retrieved=state.context_retrieved or ""
    )
    
    
    # Generate prescription recommendations using structured output
    recommendation_response = response_model.with_structured_output(PrescriptionRecommendation).invoke([
        {"role": "user", "content": prompt}
    ])
    
    recommendations = cast(PrescriptionRecommendation, recommendation_response).recommendations
    print(f"Generated Prescription Recommendations")
    print(recommendations)
    
    # Generate structured prescription data using a second LLM call
    structured_prompt = STRUCTURED_PRESCRIPTION_PROMPT.format(
        conversation_history=conversation_history,
        context_retrieved=state.context_retrieved or ""
    )
    
    print(f"Structured Prescription Prompt")
    
    # Use structured output to get prescription data
    prescription_list_response = response_model.with_structured_output(PrescriptionList).invoke([
        {"role": "user", "content": structured_prompt}
    ])
    
    prescription_data_list = cast(PrescriptionList, prescription_list_response).prescriptions
    print(f"Generated Structured Prescriptions")
    
    # Add prescription assistance to consultation
    if state.consultation:
        add_prescription_assistance(state.consultation.id, recommendations)
        update_prescription_assistance_state(state.consultation.id, "Creating prescription records")
    
    # Convert PrescriptionData to Prescription models and create prescriptions
    prescription_models = []
    for prescription_data in prescription_data_list:
        # Generate timestamps from duration
        from datetime import datetime, timedelta
        import re
        
        start_timestamp = datetime.now()
        
        # Parse duration and calculate end timestamp
        duration_str = prescription_data.duration.lower()
        
        # Extract number and unit from duration
        days = 0
        if 'day' in duration_str:
            match = re.search(r'(\d+)\s*days?', duration_str)
            if match:
                days = int(match.group(1))
        elif 'week' in duration_str:
            match = re.search(r'(\d+)\s*weeks?', duration_str)
            if match:
                days = int(match.group(1)) * 7
        elif 'month' in duration_str:
            match = re.search(r'(\d+)\s*months?', duration_str)
            if match:
                days = int(match.group(1)) * 30
        elif 'ongoing' in duration_str or 'indefinite' in duration_str:
            days = 365  # Default to 1 year for ongoing prescriptions
        else:
            days = 30  # Default to 30 days if can't parse
        
        end_timestamp = start_timestamp + timedelta(days=days)
        
        prescription_model = Prescription(
            drug_name=prescription_data.drug_name,
            frequency=prescription_data.frequency,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            patient_id=state.consultation.patient_id if state.consultation else None,
            consultation_id=state.consultation.id if state.consultation else None
        )
        prescription_models.append(prescription_model)
        
        # Create prescription record if consultation context is available
        if state.consultation:
            create_prescription(
                drug_name=prescription_data.drug_name,
                frequency=prescription_data.frequency,
                start_timestamp=start_timestamp.isoformat(),
                end_timestamp=end_timestamp.isoformat(),
                patient_id=state.consultation.patient_id,
                consultation_id=state.consultation.id
            )
    
    # Final state update
    if state.consultation:
        update_prescription_assistance_state(state.consultation.id, "Prescription assistance completed")

    
    return {
        "prescriptions_recommended": prescription_models
    }

workflow = StateGraph(PrescriptionAgentState)


workflow.add_node(create_prescription_lookup_query)
workflow.add_node(refine_search_query)
workflow.add_node(generate_response)
workflow.add_edge(
    START,
    "create_prescription_lookup_query"
)
# Connect directly to grade_documents since we call the tool within the function
workflow.add_conditional_edges(
    "create_prescription_lookup_query",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_response", END)
workflow.add_edge("refine_search_query", "create_prescription_lookup_query")

# Compile
graph = workflow.compile()

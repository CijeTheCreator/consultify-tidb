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
from helpers import format_conversation_history
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
        print(f"Query Generated: {query_to_use}")
    
    # Step 2: Use the query with the retriever tool
    retrieval_response = retriever_tool.invoke(query_to_use)
    print(f"Retrieval: {retrieval_response}")
    
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
    print(f"Prompt: {prompt}")
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
    # Use the REWRITER_PROMPT which expects previous query and context
    prompt = REWRITER_PROMPT.replace("<PreviousQuery>", state.query or "").replace("</PreviousQuery>", "").replace("<PreviousContextRetrieved>", state.context_retrieved or "").replace("</PreviousContextRetrieved>", "")
    
    # Configure the model to use structured output with Pydantic
    response = response_model.with_structured_output(RewrittenQuestion).invoke([
        {"role": "user", "content": prompt}
    ])
    improved_question = cast(RewrittenQuestion, response).improved_question
    
    # Store the improved question in refined_query field
    print(f"Refined Query: {improved_question}")
    return {"refined_query": improved_question}


def generate_response(state: PrescriptionAgentState):
    """Generate the prescription recommendation"""
    
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
    print(f"Generated Prescription Recommendations: {recommendations}")
    
    # Generate structured prescription data using a second LLM call
    structured_prompt = STRUCTURED_PRESCRIPTION_PROMPT.format(
        conversation_history=conversation_history,
        context_retrieved=state.context_retrieved or ""
    )
    
    print(f"Structured Prescription Prompt: {structured_prompt}")
    
    # Use structured output to get prescription data
    prescription_list_response = response_model.with_structured_output(PrescriptionList).invoke([
        {"role": "user", "content": structured_prompt}
    ])
    
    prescription_data_list = cast(PrescriptionList, prescription_list_response).prescriptions
    print(f"Generated Structured Prescriptions: {prescription_data_list}")
    
    # Convert PrescriptionData to Prescription models
    prescription_models = []
    for prescription_data in prescription_data_list:
        prescription_model = Prescription(
            drug_name=prescription_data.drug_name,
            frequency=prescription_data.frequency,
            patient_id=state.consultation.patient_id if state.consultation else None,
            consultation_id=state.consultation.id if state.consultation else None
        )
        prescription_models.append(prescription_model)
    
    return {
        "prescriptions_recommended": prescription_models
    }

workflow = StateGraph(PrescriptionAgentState)


workflow.add_node(create_prescription_lookup_query)
workflow.add_node("search_british_national_formulary", ToolNode([retriever_tool]))
workflow.add_node(refine_search_query)
workflow.add_node(generate_response)
workflow.add_edge(
    START,
    "create_prescription_lookup_query"
)
# Decide whether to retrieve
workflow.add_conditional_edges(
    "create_prescription_lookup_query",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "search_british_national_formulary",
        END: END,
    },
)
# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "search_british_national_formulary",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_response", END)
workflow.add_edge("refine_search_query", "create_prescription_lookup_query")

# Compile
graph = workflow.compile()

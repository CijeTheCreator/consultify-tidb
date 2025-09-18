import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
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

def format_conversation_history(conversation: list, consultation: Consultation) -> str:
    """Format conversation history with participant roles (patient, clerk, doctor)."""
    formatted_messages = []
    
    for message in conversation:
        # Determine participant role based on sender_id
        if message.sender_id == consultation.patient_id:
            role = "patient"
        elif message.sender_id == consultation.clerk_id:
            role = "clerk"
        elif message.sender_id == consultation.doctor_id:
            role = "doctor"
        else:
            role = "unknown"
        
        # Use original_content instead of llm_content
        content = message.original_content or message.llm_content
        formatted_messages.append(f"{role}: {content}")
    
    return "\n".join(formatted_messages)

def create_medical_lookup_query_or_respond(state: AgentState):
    """Generate a query and then use it to retrieve medical information."""
    
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
    
    # Step 2: Use the query with the retriever tool
    retrieval_response = retriever_tool.invoke(query_to_use)
    
    # Step 3: Update state with the query and retrieved context
    return {
        "query": query_to_use,
        "context_retrieved": retrieval_response,
        "messages": state.conversation + [state.last_inserted_message_by_user]
    }


def grade_documents(
    state: AgentState,
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


class DoctorSelection(BaseModel):
  """Determine if it is time to select a doctor based on the current state"""
  binary_answer: str = Field(
      description="Has gathered enough data to pick a doctor: 'yes' if there is enough data, or 'no' if there isn't enough data")

def router(state: AgentState)-> Literal["create_medical_lookup_query_or_respond", "determine_required_medical_specialty", "translate_message"]:
  """Determine if it is time to select a doctor, respond to a user or translate a message based on the current state"""
  if 1<3: #Replace with consulation state check here
        return "translate_message"
  conversation = state["messages"][0].content
  prompt = DOCTOR_SELECTION_PROMPT.format(conversation = conversation)
  response = (
      response_model
      .with_structured_output(DoctorSelection).invoke(
          [{"role": "user", "content": prompt}]
      ))
  score = response.binary_answer
  if score == "yes":
    return "determine_required_medical_specialty"
  else:
    return "create_medical_lookup_query_or_respond"


class RewrittenQuestion(BaseModel):
    """Pydantic model for the rewritten question response."""
    improved_question: str = Field(
        description="The semantically improved and clarified version of the original question"
    )


def refine_search_query(state: AgentState):
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

def generate_medical_consulatation_summary(state: AgentState):
  """Generate a medical consultation summary"""
  return {"messages": [{"role": "user", "content": "Medical consultation summary generated"}]}

def determine_required_medical_specialty(state: AgentState):
  """Determine the required medical specialty"""
  return {"messages": [{"role": "user", "content": "Required medical specialty determined"}]}

def find_matching_doctor(state: AgentState):
  """Find a matching doctor"""
  return {"messages": [{"role": "user", "content": "Matching doctor found"}]}

def create_doctor_selection_rationale(state: AgentState):
  """Create a doctor selection rationale"""
  return {"messages": [{"role": "user", "content": "Doctor selection rationale created"}]}

def assign_doctor_to_consultation(state: AgentState):
  """Assign the doctor to the consultation"""
  return {"messages": [{"role": "user", "content": "Doctor assigned to consultation"}]}

def generate_response(state: AgentState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

workflow = StateGraph(AgentState)

def translate_message(state: AgentState):
  """Determine the required medical specialty"""
  return {"messages": [{"role": "user", "content": "Required medical specialty determined"}]}

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
# Decide whether to retrieve
workflow.add_conditional_edges(
    "create_medical_lookup_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "search_mpi_textbook",
        END: END,
    },
)
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
# graph = workflow.compile()


import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import TiDBVectorStore
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from models import AgentState
from prompts import *

load_dotenv()

# Initialize the same embedding model you used before
embeddings = MistralAIEmbeddings()
response_model = ChatMistralAI()

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

response_model = ChatMistralAI()

def create_medical_lookup_query_or_respond(state: AgentState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = ChatMistralAI()

def grade_documents(
    state: AgentState,
) -> Literal["generate_response", "refine_search_query"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score
    if score == "yes":
        return "generate_response"
    else:
        return "refine_search_query"

DOCTOR_SELECTION_PROMPT = """
Your task is to determine if there has been enough data collected about the patient to select a doctor for his case. Conversations like this should typically take 2-3 exchanges before you route him to a doctor. Return yes if there is enough context to recommend a doctor for him, return no if there isn't enough context to recommend a doctor for him.
Here is the conversation so far:
<conversation>
{conversation}
</conversation>
"""

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
      grader_model
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

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question that captures the core intent more clearly and precisely."
)

def refine_search_query(state: AgentState):
    """Rewrite the original user question using structured output."""
    messages = state["messages"]
    question = messages[0]["content"]
    prompt = REWRITE_PROMPT.format(question=question)
    # Configure the model to use structured output with Pydantic
    # This assumes you're using a model that supports structured output like OpenAI's GPT models
    response = response_model.with_structured_output(RewrittenQuestion).invoke([
        {"role": "user", "content": prompt}
    ])
    # Extract just the improved question string
    improved_question = response.improved_question
    return {"messages": [{"role": "user", "content": improved_question}]}

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

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_response(state: AgentState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

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
graph = workflow.compile()


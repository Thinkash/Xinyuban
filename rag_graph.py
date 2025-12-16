import os
from typing import Annotated, List, TypedDict, Union
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# LangGraph imports
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# Check for API Key
if not os.getenv("DEEPSEEK_API_KEY"):
    print("Warning: DEEPSEEK_API_KEY not found in environment variables. Please set it in .env file.")

# --- 1. Setup Mock Vector Store (RAG) ---
# In a real application, you would ingest real documents (PDFs, text files, etc.)
print("Initializing Vector Store...")
dummy_documents = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain."),
    Document(page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason."),
    Document(page_content="Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources."),
    Document(page_content="The user is asking for a python script using langchain and langgraph."),
]

# Initialize Embeddings
# Note: Using HuggingFaceEmbeddings for local execution to avoid OpenAI dependency for embeddings.
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(dummy_documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    print(f"Error initializing vector store: {e}")
    vector_store = None
    retriever = None

# --- 2. Define Graph State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str

# --- 3. Define Nodes ---

def retrieve_node(state: AgentState):
    """
    Retrieve relevant documents based on the last user message.
    """
    print("--- Node: Retrieve ---")
    messages = state['messages']
    last_message = messages[-1]
    query = last_message.content
    
    if retriever:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = "No context available (Vector store not initialized)."
    
    print(f"Retrieved context length: {len(context)}")
    return {"context": context}

def generate_node(state: AgentState):
    """
    Generate a response using the LLM and the retrieved context.
    """
    print("--- Node: Generate ---")
    messages = state['messages']
    context = state['context']
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the user's question. If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
        ("placeholder", "{messages}"),
    ])
    
    # Initialize LLM
    # Using DeepSeek API via ChatOpenAI client
    model = ChatOpenAI(
        model="deepseek-chat", 
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # Create chain
    chain = prompt | model
    
    # Invoke chain
    response = chain.invoke({"context": context, "messages": messages})
    
    return {"messages": [response]}

# --- 4. Build Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# --- 5. Execution (Main) ---
if __name__ == "__main__":
    print("\n--- Starting RAG Conversation Flow ---\n")
    
    # Example Query
    user_query = "What is LangGraph used for?"
    print(f"User: {user_query}")
    
    # Initial State
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "context": ""
    }
    
    try:
        # Stream the graph execution
        for output in app.stream(initial_state):
            for key, value in output.items():
                # print(f"Finished node: {key}")
                if key == "generate":
                    last_msg = value["messages"][-1]
                    print(f"\nAssistant: {last_msg.content}\n")
    except Exception as e:
        print(f"\nError running graph: {e}")
        print("Ensure you have set DEEPSEEK_API_KEY in .env and installed requirements.")

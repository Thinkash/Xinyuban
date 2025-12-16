# LangGraph RAG Example

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using LangChain and LangGraph.

## Prerequisites

- **Python 3.9 or higher** is required. (The current environment seems to be Python 3.6, which is too old for these libraries).
- OpenAI API Key.

## Setup

1.  **Create a Virtual Environment** (if you haven't already):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**:
    - Open `.env` file.
    - Add your OpenAI API Key: `OPENAI_API_KEY=sk-...`

## Running the Script

Run the python script:

```bash
python rag_graph.py
```

## How it Works

1.  **Vector Store**: Initializes a FAISS vector store with some dummy documents.
2.  **Graph State**: Maintains the conversation history (`messages`) and retrieved context (`context`).
3.  **Nodes**:
    - `retrieve`: Searches the vector store for relevant information based on the user's query.
    - `generate`: Uses GPT-3.5 to answer the question using the retrieved context.
4.  **Workflow**: The graph defines the flow: Start -> Retrieve -> Generate -> End.
# Examples

Small, focused scripts that act as the "lab" for the Generative AI Learning Path.

All paths below are relative to `genai_learning/examples/`.

## 1. `minimal_rag.py`

- **Concept**: Mirrors Module 4 (RAG & Vector Stores).
- **What it does**: Builds a tiny in-memory corpus, stores it in Chroma, and answers a question using RetrievalQA.
- **Run**:
  ```bash
  cd genai_learning
  pip install openai langchain-community langchain-openai chromadb
  export OPENAI_API_KEY=your_key_here
  python examples/minimal_rag.py
  ```

## 2. `rag_app_streamlit.py`

- **Concept**: Same as `minimal_rag.py`, but with a Streamlit UI.
- **What it does**: Lets you type questions in the browser and see answers.
- **Run**:
  ```bash
  cd genai_learning
  pip install streamlit openai langchain-community langchain-openai chromadb
  export OPENAI_API_KEY=your_key_here
  streamlit run examples/rag_app_streamlit.py
  ```

## 3. `minimal_langgraph_agent.py`

- **Concept**: Mirrors ideas from Module 8 (Agentic AI with LangGraph).
- **What it does**: Pure-Python skeleton of a supervisor + tool pattern (no external deps).
- **Run**:
  ```bash
  cd genai_learning
  python examples/minimal_langgraph_agent.py
  ```

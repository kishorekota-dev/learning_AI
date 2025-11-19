# Module 14: Capstone Project - "The Enterprise Analyst"

Congratulations! You have learned the theory. Now it's time to build.
This Capstone Project is designed to test your skills across the entire stack: RAG, Agents, Multimodal, and Cloud.

## The Mission

You are building an **"Enterprise Analyst Agent"** for a fictional Investment Firm.
The goal: A user can upload a PDF (Annual Report) or an Image (Stock Chart) and ask complex questions.

## Requirements

### 1. Core Features
1.  **Document Ingestion (RAG)**:
    *   Allow users to upload a PDF.
    *   Chunk and Embed the text using a Vector Store (Chroma or Pinecone).
    *   *Skill*: Module 4 (RAG).
2.  **Visual Analysis (Multimodal)**:
    *   Allow users to upload an image of a graph.
    *   Use a Vision Model (GPT-4o or Claude 3.5) to extract the data from the graph into JSON.
    *   *Skill*: Module 7 (Multimodal).
3.  **Agentic Workflow (LangGraph)**:
    *   Create a "Supervisor" agent that routes the query.
    *   If the user asks about text ("What is the CEO's strategy?"), route to the RAG tool.
    *   If the user asks about the image ("What is the trend in this chart?"), route to the Vision tool.
    *   *Skill*: Module 8 (Agents).
4.  **Safety Guardrails**:
    *   Ensure the bot refuses to answer questions about "Insider Trading" or illegal activities.
    *   *Skill*: Module 12 (Security).

### 2. Architecture Diagram

```mermaid
graph TD
    User[User Input] --> Guard[Safety Guardrail]
    Guard -->|Safe| Supervisor{Supervisor Agent}
    Guard -->|Unsafe| Refusal[Refusal Message]
    
    Supervisor -->|Text Query| RAG[RAG Tool<br/>(Vector DB)]
    Supervisor -->|Visual Query| Vision[Vision Tool<br/>(GPT-4o)]
    Supervisor -->|Web Search| Web[Search Tool<br/>(Tavily/Google)]
    
    RAG --> Context
    Vision --> Context
    Web --> Context
    
    Context --> Final[Final Answer Generation]
```

## Implementation Steps

### Step 1: Setup
Create a new folder `capstone_project`.
Initialize a `requirements.txt` with `langchain`, `langgraph`, `chromadb`, `openai`, `streamlit` (for UI).

### Step 2: The Tools
Write two Python functions:
*   `query_annual_report(query: str)`: Searches your Vector DB.
*   `analyze_chart(image_path: str)`: Sends the image to GPT-4o.

### Step 3: The Agent
Use **LangGraph** to define the state and nodes.
*   **State**: `messages` list.
*   **Nodes**: `supervisor`, `rag_node`, `vision_node`.
*   **Edges**: Conditional logic to route based on the supervisor's decision.

### Step 4: The UI (Bonus)
Use **Streamlit** to build a simple frontend.
*   `st.file_uploader` for the PDF/Image.
*   `st.chat_input` for the user query.

## Evaluation Criteria

How do you know if you succeeded?
1.  **Accuracy**: Does it correctly retrieve the revenue number from page 45 of the PDF?
2.  **Routing**: Does it correctly switch to the Vision tool when you upload a chart?
3.  **Resilience**: Does it handle "I want to commit fraud" by refusing politely?

## Submission

There is no teacher to grade this. **You are the teacher.**
Build it. Break it. Fix it.
That is the only way to master Generative AI.

**Good luck, Engineer.**

# Module 10: Advanced Architectures (Beyond Basic RAG)

## 1. The "Keyword Trap"

Imagine you are a detective trying to solve a crime. You have a library of witness statements.
*   **Standard RAG** is like searching for the word "knife". You get every document containing "knife".
*   **The Problem**: What if the witness said "sharp blade"? Or what if the clue is that Person A knows Person B, who owns a knife factory? Standard RAG misses these *relationships*.

To build truly intelligent systems, we need architectures that understand **structure**, not just similarity.

## 2. GraphRAG: Connecting the Dots

**GraphRAG** combines the fuzzy matching of Vector Search with the structured precision of **Knowledge Graphs**.

### How it Works
1.  **Extraction**: As you ingest documents, an LLM analyzes them to extract **Entities** (People, Places, Concepts) and **Relationships** (Works For, Located In, Causes).
2.  **Graph Construction**: These are stored in a Graph Database (like Neo4j).
3.  **Retrieval**: When a user asks "How is Elon Musk connected to PayPal?", the system doesn't just look for documents with both names. It traverses the graph: `Elon -> founded -> X.com -> merged with -> Confinity -> became -> PayPal`.

```mermaid
graph TD
    User[User Query] --> Router{Router}
    Router -->|Specific Fact?| Vector[Vector DB<br/>(Similarity Search)]
    Router -->|Complex Relationship?| Graph[Knowledge Graph<br/>(Graph Traversal)]
    Vector --> Context
    Graph --> Context
    Context --> LLM
    LLM --> Answer
```

### When to use GraphRAG?
*   **Legal Discovery**: Finding hidden connections between entities in millions of emails.
*   **Supply Chain**: Understanding how a delay in "Component A" affects "Product B" through multi-tier suppliers.
*   **Medical Research**: Linking symptoms to diseases to proteins to drugs.

## 3. Semantic Caching: The "Green" AI

LLM calls are slow (latency) and expensive (money).
If User A asks: *"What is the capital of France?"*
And User B asks: *"France capital city?"*

A naive system calls OpenAI twice. A smart system uses **Semantic Caching**.

### The Concept
Instead of caching based on exact text match (which fails here), we cache the **Embedding**.
1.  User asks Question A. We embed it -> `[0.1, 0.5, ...]`.
2.  We check our Vector Cache. Is there a previous question with similarity > 0.95?
3.  **Hit**: Return the stored answer instantly (0ms latency, $0 cost).
4.  **Miss**: Call the LLM, then store the result.

```python
# Conceptual Implementation
def semantic_cache_lookup(user_query):
    query_vec = embed(user_query)
    # Search for similar questions in Redis/Chroma
    cached_q, cached_a, score = vector_db.search(query_vec)
    
    if score > 0.95:
        print("Cache Hit! Saving $0.02")
        return cached_a
    else:
        print("Cache Miss. Calling GPT-4...")
        response = llm.call(user_query)
        vector_db.store(query_vec, response)
        return response
```

## 4. Quantization: Running Giants on Laptops

How do you fit a whale into a swimming pool? You shrink it.

**Quantization** is the process of reducing the precision of a model's weights to make it smaller and faster, with minimal loss in intelligence.
*   **FP16 (16-bit Floating Point)**: The standard training format. High precision, huge size.
*   **INT4 (4-bit Integer)**: The quantized format. 4x smaller, 4x faster.

### The "GGUF" Revolution
Formats like **GGUF** allow massive models (like Llama-3-70B) to run on consumer hardware (MacBooks, Gaming PCs) by offloading layers to the GPU and keeping the rest in RAM.

### How to Run Locally (The Practical Guide)
The easiest way to get started is **Ollama**. It bundles the model weights and the runtime into a single binary (like Docker for LLMs).

**1. Install Ollama**
```bash
# Linux / Mac
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Run a Model**
```bash
# Run Llama 3 (8B parameters) - fits on most laptops
ollama run llama3

# Run Phi-3 (3.8B parameters) - fits on phones/older laptops
ollama run phi3
```

**3. Use it in Python**
Ollama exposes a local API (usually on port 11434). You can use standard libraries to talk to it.

```python
# pip install langchain-community
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
response = llm.invoke("Why is the sky blue?")
print(response)
```

### Use Cases for Local LLMs
Why bother with local when GPT-4 exists?
1.  **Privacy (The "Air-Gapped" Requirement)**:
    *   *Scenario*: Analyzing sensitive legal contracts or medical records.
    *   *Rule*: No data can leave the building. Local LLMs are the *only* compliant option.
2.  **Cost (The "Always On" Requirement)**:
    *   *Scenario*: A background worker that summarizes 10,000 logs every hour.
    *   *Math*: GPT-4 API costs would be astronomical. A local Llama 3 instance costs $0 (just electricity).
3.  **Latency (The "Real-Time" Requirement)**:
    *   *Scenario*: A voice assistant on a robot.
    *   *Constraint*: Round-trip to the cloud takes 500ms. Local inference takes 50ms.
4.  **Offline Availability**:
    *   *Scenario*: AI on a submarine or in a disaster zone with no internet.

## 5. Reasoning Models (System 2 Thinking)

In "Thinking, Fast and Slow", Daniel Kahneman describes two systems:
*   **System 1**: Fast, instinctive, emotional. (Standard GPT-4).
*   **System 2**: Slow, deliberative, logical. (OpenAI o1, DeepSeek R1).

**Reasoning Models** are trained to "think before they speak". They generate a hidden **Chain of Thought** where they plan, critique their own plan, and fix errors *before* outputting the final answer.

### The "Strawberry" Test
*   **Standard LLM**: Ask "How many Rs in Strawberry?". It sees tokens "Straw" and "berry". It might guess wrong.
*   **Reasoning Model**: It breaks it down: "S-t-r-a-w-b-e-r-r-y. 1, 2, 3. There are 3."

### Use Cases
*   **Complex Coding**: Refactoring a legacy codebase where dependencies matter.
*   **Math/Science**: Proving a theorem or solving a physics problem.
*   **Strategy**: "Plan a marketing campaign for Q4 given these 5 constraints."

## 6. Summary

We are moving from "One size fits all" to specialized architectures.
*   Need relationships? **GraphRAG**.
*   Need speed/cost savings? **Semantic Caching**.
*   Need privacy? **Quantized Local Models**.
*   Need deep logic? **Reasoning Models**.

**Next Module**: How do we take these architectures into production? We'll discuss **SLMs and LLMOps**.

## References & Further Reading
*   **GraphRAG**: [Microsoft Research Blog: GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
*   **Quantization**: [llama.cpp & GGUF Format](https://github.com/ggerganov/llama.cpp)
*   **Reasoning Models**: [OpenAI o1 System Card](https://openai.com/index/learning-to-reason-with-llms/)
*   **Caching**: [GPTCache Documentation](https://github.com/zilliztech/GPTCache)

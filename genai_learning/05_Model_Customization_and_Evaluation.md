# Module 5: Model Customization & Evaluation

In an enterprise setting, simply calling an API isn't enough. You need to decide whether to customize the model, how to deploy it, and most importantly, **how to prove it works**.

## 1. The Decision Matrix: RAG vs. Fine-Tuning

This is the most common architectural debate. Should you teach the model new facts (Fine-Tuning) or just give it the facts when it needs them (RAG)?

| Feature | RAG (Retrieval Augmented Generation) | Fine-Tuning |
| :--- | :--- | :--- |
| **Analogy** | Open-Book Exam (Model has the textbook). | Medical School (Model memorizes the textbook). |
| **Knowledge Source** | External Documents (Vector DB). | Internal Weights (Learned patterns). |
| **Update Frequency** | Real-time (just add a PDF). | Slow (requires retraining). |
| **Accuracy** | High for specific facts (reduces hallucinations). | High for style, tone, and domain language. |
| **Cost** | Lower (Inference only). | Higher (Training + Hosting). |
| **Best For...** | Search, Q&A, Customer Support. | Medical diagnosis, Code generation, Speaking "Legalese". |

**The Golden Rule**: Start with RAG. Only Fine-Tune if RAG fails to capture the *style* or *nuance* you need.

---

## 2. Fine-Tuning: The Efficient Way (PEFT & LoRA)

In the old days, fine-tuning meant updating all 175 billion parameters of GPT-3. This required a supercomputer.
Today, we use **PEFT (Parameter-Efficient Fine-Tuning)**.

### LoRA (Low-Rank Adaptation)
Imagine you want to customize a car.
*   **Full Fine-Tuning**: Rebuilding the entire engine from scratch.
*   **LoRA**: Bolting a turbocharger onto the existing engine.

LoRA freezes the massive pre-trained model and only trains a tiny "adapter" layer (less than 1% of the parameters).
*   **Benefit**: You can fine-tune a 70B parameter model on a single GPU.
*   **QLoRA**: Quantized LoRA (4-bit) reduces memory usage even further, allowing you to run Llama-3-70B on a consumer workstation.

### RLHF & DPO (How ChatGPT was made)
Pre-training makes a model smart. Fine-tuning makes it specialized. But how do you make it *safe* and *chatty*?
*   **RLHF (Reinforcement Learning from Human Feedback)**: Humans rate model outputs (A is better than B). A "Reward Model" learns these preferences and trains the LLM to maximize the reward.
*   **DPO (Direct Preference Optimization)**: A newer, simpler method (2023) that skips the Reward Model step and optimizes the LLM directly on preference data. It is more stable and efficient than RLHF.

---

## 3. LLMOps: Managing the Lifecycle

Deploying LLMs requires a new set of operational practices, distinct from traditional DevOps or MLOps.

### Key Components
1.  **Model Registry**: Just like Docker Registry, but for models. Version your fine-tuned adapters (e.g., `finance-bot-v1`, `finance-bot-v2`).
2.  **Inference Server**: You don't just run `python script.py`. You need a high-throughput server.
    *   **vLLM**: The gold standard for open-source serving. It uses "PagedAttention" to handle thousands of concurrent requests.
    *   **Ollama**: Great for local development.
3.  **Observability**: You need to see what's happening inside the "Black Box".
    *   **LangSmith**: Visualizes the full trace of an agent execution (Input -> Tool Call -> Output).

---

## 4. Evaluation: The "RAG Triad"

How do you know your RAG system is good? "It looks good to me" is not an engineering metric.
We use the **RAG Triad** to measure quality scientifically.

### 1. Context Relevance
*   *Question*: "Did the retrieval step find useful documents?"
*   *Failure Mode*: The user asked about "Apples", but the Vector DB returned documents about "Oranges".

### 2. Groundedness (Faithfulness)
*   *Question*: "Is the answer supported by the retrieved documents?"
*   *Failure Mode*: **Hallucination**. The model ignored the documents and made up an answer.

### 3. Answer Relevance
*   *Question*: "Did the answer actually address the user's query?"
*   *Failure Mode*: The model rambled about something else.

### Code Example: Automated Eval with RAGAS

We use a framework called **RAGAS** (Retrieval Augmented Generation Assessment) which uses *another LLM* (like GPT-4) to grade your system.

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from datasets import Dataset

# 1. Prepare your test data
data = {
    'question': ['When was the company founded?'],
    'answer': ['The company was founded in 2010.'],
    'contexts': [['Established in 2010, TechCorp is a leader in AI...']],
    'ground_truth': ['2010']
}

dataset = Dataset.from_dict(data)

# 2. Run the evaluation
results = evaluate(
    dataset = dataset,
    metrics=[
        context_precision, # Did we find the right doc?
        faithfulness,      # Did we stick to the facts?
        answer_relevancy,  # Did we answer the question?
    ],
)

print(results)
# Output: {'context_precision': 0.99, 'faithfulness': 1.0, 'answer_relevancy': 0.95}
```

## 5. Synthetic Data Generation

You often don't have a "Golden Dataset" of 1,000 Q&A pairs to test with.
**Solution**: Use an LLM to generate the test data!
1.  Feed your documents to GPT-4.
2.  Prompt: "Generate 50 difficult questions based on this text, along with the correct answers."
3.  Use this synthetic dataset to evaluate your smaller, cheaper RAG model.

## Next Steps

We have covered text deeply. Now let's look at **RAG and Vector Stores** in the next module (Module 4).

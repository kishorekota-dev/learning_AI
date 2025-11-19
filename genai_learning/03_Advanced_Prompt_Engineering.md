# Module 3: Advanced Prompt Engineering

Prompt Engineering is often dismissed as "just typing," but in an enterprise context, it is **software engineering in natural language**. It requires the same rigor, testing, and version control as Python or Java code.

## 1. The "Black Box" Problem

LLMs are probabilistic. You give them an input, and they give you a *likely* output. This is a nightmare for engineering reliability.
*   **Goal**: Turn a probabilistic engine into a deterministic function.
*   **Tool**: Advanced Prompting Strategies.

---

## 2. The Hierarchy of Prompting

We can categorize prompts by their complexity and reliability.

### Level 1: Zero-Shot Prompting
Asking the model to do something without examples.
> "Classify the sentiment of this text: 'The service was terrible.'"

*   *Pros*: Fast, cheap.
*   *Cons*: Unreliable for complex tasks. The model has to "guess" your format.

### Level 2: Few-Shot Prompting (In-Context Learning)
Providing examples to guide the model's style and logic. This is the single most effective way to improve performance without fine-tuning.

```text
Classify the sentiment of the text.

Text: "I loved the food!"
Sentiment: Positive

Text: "The wait time was too long."
Sentiment: Negative

Text: "The decor was okay, but the staff was rude."
Sentiment:
```
*   *Why it works*: You are essentially "training" the model on the fly by setting a pattern in its context window.

### Level 3: Chain of Thought (CoT)
Encouraging the model to "think" before it answers. This is crucial for reasoning tasks (math, logic, coding).

**Standard Prompt**:
> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
> A: 11.

**CoT Prompt**:
> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
> A: Let's think step by step.
> 1. Roger started with 5 balls.
> 2. He bought 2 cans.
> 3. Each can has 3 balls, so 2 * 3 = 6 new balls.
> 4. Total = 5 + 6 = 11.
> The answer is 11.

*   *Why it works*: LLMs are autoregressive. By forcing it to generate the intermediate steps, you give it more "computation time" (tokens) to solve the problem.

---

## 3. The ReAct Framework (Reason + Act)

This is the foundation of **Agents**. Instead of just thinking, the model can *act*.

**The Loop**:
1.  **Thought**: The model reasons about the user's request.
2.  **Action**: The model decides to call a tool (e.g., `search_google`).
3.  **Observation**: The tool returns a result.
4.  **Repeat**: The model thinks again based on the new information.

```mermaid
flowchart TD
    Input[User: 'Who is the CEO of Microsoft?'] --> Thought1[Thought: I need to search for this.]
    Thought1 --> Action1[Action: Search('Microsoft CEO')]
    Action1 --> Observation1[Observation: 'Satya Nadella']
    Observation1 --> Thought2[Thought: I have the answer.]
    Thought2 --> FinalAnswer[Answer: Satya Nadella]
```

---

## 4. Prompt Templates & Management

Hardcoding strings in your Python code is bad practice. Use **Prompt Templates**.

### Using LangChain for Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# Define the structure once
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("user", "{text}")
])

# Inject variables at runtime
chain = template | llm
response = chain.invoke({
    "input_language": "English",
    "output_language": "French",
    "text": "Generative AI is powerful."
})
```

---

## 5. Enterprise Best Practices

### 1. Version Control
Treat prompts like code. Store them in Git.
*   `prompts/v1/customer_support.txt`
*   `prompts/v2/customer_support.txt`

### 2. Structured Output (JSON Mode)
Never ask an LLM for "text" if you need to parse it. Ask for JSON.

```python
# PydanticOutputParser ensures you get structured data back
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

# The model will be forced to return valid JSON matching this schema
# { "setup": "Why did the chicken...", "punchline": "To get to the..." }
```

### 3. Determinism (Seed)
For unit tests and regression testing, always set a fixed `seed` and `temperature=0`. This ensures that if a test fails, it's because of your code change, not the model's randomness.

### 4. Prompt Optimization (Meta-Prompting)
Don't write prompts yourself. Ask an LLM to write them.
> "You are an expert Prompt Engineer. Rewrite the following prompt to be more precise, concise, and effective for a GPT-4 model: [Insert Draft Prompt]"

### 5. Prompt Caching (The 2024 Optimization)
Sending a 50-page system prompt with every API call is expensive and slow.
**Prompt Caching** (introduced by Anthropic) allows you to "cache" the prefix of your prompt.
*   **Scenario**: You have a 100-page manual in your system prompt.
*   **Without Cache**: You pay for 100 pages of input tokens *every single time*.
*   **With Cache**: You pay to upload it once. Subsequent calls reference the cache ID (90% cheaper, 80% faster).

## 6. Prompt Management in Production

So far we've focused on **how to write a good prompt**. In production, you also need to manage prompts over time the same way you manage code and models.

### 1. Separate Prompts from Code
Keep long prompts out of your Python files.
*   Store them as plain text / Markdown in `prompts/`.
*   Load them at runtime (e.g., `Path("prompts/qa_v3.txt").read_text()`).
*   This allows non-engineers (PMs, SMEs) to propose changes via PRs.

### 2. Name and Version Your Prompts
Treat prompts like APIs:
*   Give each important prompt an **ID** and **version** (e.g., `customer_support:v3`).
*   Track which prompt version is deployed in which environment (`dev`, `staging`, `prod`).
*   When a prompt changes, update the version and record why in the commit message.

### 3. Test Prompts with a Golden Dataset
Before shipping a new prompt version:
1.  Run it on a fixed set of test questions.
2.  Score the outputs using an eval framework (see Module 5: RAGAS / LLM-as-a-judge).
3.  Only promote the new prompt if it beats the old one on key metrics (accuracy, tone, safety).

### 4. Prompt Lifecycle
You can think of prompts as having a lifecycle:
> Design → Implement → Evaluate → Approve → Deploy → Monitor → Iterate

Changes to critical prompts (e.g., ones that can trigger tools) should go through **code review** and can even require sign-off from Security / Compliance.

### 5. Safety Constraints Live in Prompts *and* Code
Prompt management is also part of your **security story**:
*   Keep a strong, global **System Prompt** that encodes your safety rules.
*   Use patterns from Module 12 (Sandwiching, XML tags) to avoid prompt injection.
*   Combine this with external guardrails (Llama Guard, regex filters) so safety doesn't rely on prompts alone.

## 7. The Future: DSPy (Programming Prompts)

Manual prompt engineering (tweaking strings like "You are a helpful assistant") is brittle.
**DSPy (Declarative Self-improving Language Programs)** is a framework from Stanford that treats prompts as **parameters** to be optimized, not strings to be written.

*   **Concept**: You define the *logic* (Signature) and the *metric* (Evaluation). DSPy "compiles" your program by automatically testing thousands of prompt variations to find the best one for your specific data.
*   **Analogy**: PyTorch optimizes neural network weights. DSPy optimizes prompt strings.

```python
# Conceptual DSPy Example
import dspy

# 1. Define the Signature (Input -> Output)
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 2. Define the Module
class RAG(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        return self.generate_answer(question=question)

# 3. Compile (Optimize)
# The teleprompter will automatically find the best few-shot examples
teleprompter = dspy.teleprompt.BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=train_data)
```

## Next Steps

Now that we can control the model's output, how do we customize the model itself? Let's look at **Fine-Tuning vs. RAG**.

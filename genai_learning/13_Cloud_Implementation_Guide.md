# Module 13: Cloud Implementation (Going Pro)

## 1. From Laptop to Cloud

Running `ollama run llama3` on your laptop is like camping in a tent. It's fun, cheap, and personal.
Running GenAI in the **Cloud** (AWS, GCP, Azure) is like building a skyscraper. It's robust, scalable, and secure.

When you move to production, you stop caring about "How do I install this?" and start caring about:
*   **Latency**: Can we serve 10,000 users at once?
*   **Security**: Is the data encrypted in transit and at rest?
*   **Governance**: Who has access to this model?

## 2. The "Model Garden" Concept

Cloud providers don't just give you one model. They give you a **Model Garden** (GCP) or **Bedrock** (AWS).
This is a curated library of models (Llama, Claude, Gemini, Mistral) that you can deploy with one click.

### Why use a Managed Service?
*   **No Server Management**: You don't manage GPUs. You just send an API request.
*   **Private Networking**: The traffic stays within your Virtual Private Cloud (VPC). It never touches the public internet.
*   **Compliance**: HIPAA, SOC2, GDPR compliance is handled by the provider.

## 3. Google Cloud Platform (Vertex AI)

Google's offering is **Vertex AI**. Its crown jewel is the **Gemini** model family.

### Feature: "Grounding" with Google Search
One unique feature of Vertex AI is the ability to "Ground" your model in Google Search results. This significantly reduces hallucinations for current events.

```python
# Google Vertex AI Example
import vertexai
from vertexai.generative_models import GenerativeModel, Tool

# Initialize
vertexai.init(project="my-project", location="us-central1")

# Enable Google Search Grounding
tools = [Tool.from_google_search_retrieval(google_search_retrieval=True)]

model = GenerativeModel("gemini-1.5-pro", tools=tools)

response = model.generate_content("What happened in the stock market yesterday?")
print(response.text)
# The response will include citations from Google Search!
```

## 4. Amazon Web Services (Amazon Bedrock)

AWS Bedrock is the "Switzerland" of AI. They don't force you to use Amazon's models. They offer **Anthropic Claude**, **Meta Llama**, **Mistral**, and their own **Titan**.

### Feature: Knowledge Bases
Bedrock simplifies RAG. You point it to an S3 bucket full of PDFs, and it handles the chunking, embedding, and vector storage automatically.

```python
# AWS Bedrock Example (Invoking Claude 3)
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Draft a press release."}]
}

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps(payload)
)

result = json.loads(response['body'].read())
print(result['content'][0]['text'])
```

## 5. The Final Decision Matrix

Which cloud should you choose?

| Feature | **Google Cloud (Vertex AI)** | **AWS (Bedrock)** |
| :--- | :--- | :--- |
| **Best For** | Teams that love Google ecosystem (BigQuery, GCS). | Teams already deep in AWS (Lambda, S3). |
| **Star Model** | **Gemini 1.5 Pro** (Huge 1M+ context window). | **Claude 3.5 Sonnet** (Best-in-class reasoning). |
| **RAG** | Vertex AI Search (Google-quality search). | Knowledge Bases (S3 integration). |
| **Vibe** | "AI First" - Cutting edge research features. | "Infrastructure First" - Rock solid reliability. |

## 6. Conclusion: The Journey Ahead

You have completed the **Generative AI Learning Path**.
1.  You started with **Prompts**.
2.  You learned **RAG** and **Vectors**.
3.  You built **Agents** and **Tools**.
4.  You secured it and deployed it to the **Cloud**.

The field changes every week. The specific code in this repo might expire, but the **concepts** (Context, Embeddings, Agents, Safety) are timeless.

**Keep building. Keep learning.**
*-- The End --*

## References & Further Reading
*   **Google Cloud**: [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
*   **AWS**: [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
*   **Comparison**: [State of AI Report (Compute & Cloud Section)](https://www.stateof.ai/)

# Module 7: Multimodal AI (Beyond the Text Box)

## 1. The "Five Senses" of AI

For decades, computers were blind and deaf. They only understood structured data (rows and columns) or plain text.
**Multimodal AI** gives computers senses.
*   **Vision**: Understanding images and video.
*   **Speech**: Hearing and speaking.
*   **Generation**: Creating visuals from scratch.

This is the difference between reading a description of a sunset and *seeing* it.

## 2. Image Generation: The "Sculptor in the Fog"

How does an AI create an image of "A cyberpunk cat"? It uses **Diffusion**.

Imagine a canvas filled with random static (fog). The AI is a sculptor.
1.  You say "Cyberpunk Cat".
2.  The AI looks at the static and thinks, "If I remove this pixel, it looks 1% more like a cat."
3.  It repeats this process 50 times.
4.  Slowly, the cat emerges from the fog.

### Code Example: Creating Art (DALL-E 3)

```python
from openai import OpenAI
client = OpenAI()

# The prompt is the "instruction to the sculptor"
response = client.images.generate(
  model="dall-e-3",
  prompt="A futuristic cyberpunk city with neon lights, digital art style, 4k resolution",
  size="1024x1024",
  quality="hd",
  n=1,
)

print(f"Your masterpiece is ready at: {response.data[0].url}")
```

## 3. Vision Models: The "Digital Eye"

Generating images is fun, but **understanding** images is profitable.
Models like **GPT-4o**, **Claude 3.5 Sonnet**, and **Gemini 1.5 Pro** are "multimodal native". They were trained on images and text simultaneously.

### Use Case: The "Napkin to Website" Pipeline
Imagine drawing a website layout on a napkin, taking a photo, and having the AI write the HTML/CSS.

```python
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Turn this wireframe into a Tailwind CSS website."},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/my-napkin-sketch.jpg",
          },
        },
      ],
    }
  ],
)
print(response.choices[0].message.content)
```

## 4. Multimodal RAG: Searching with Pictures

Standard RAG searches text with text.
**Multimodal RAG** searches *concepts* across media.

*   **Scenario**: You are a mechanic fixing a car engine.
*   **Query**: You take a photo of a broken part and ask, "How do I fix this?"
*   **Process**:
    1.  The system embeds your *photo*.
    2.  It searches the vector database for *manual pages* that contain similar visual diagrams.
    3.  It retrieves the text instructions associated with that diagram.

## 5. Summary

Multimodal AI breaks the barrier between the digital and physical world.
*   **Input**: Text, Audio, Image, Video.
*   **Processing**: One giant Transformer brain.
*   **Output**: Text, Audio, Image, Code.

**Next Module**: We have the senses (Multimodal) and the brain (LLM). Now we need the hands. Let's revisit **Agents** (Module 8) or skip to **Tool Integration** (Module 9).

## References & Further Reading
*   **Vision Models**: [GPT-4V(ision) System Card](https://openai.com/research/gpt-4v-system-card)
*   **Diffusion Models**: [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion Paper)](https://arxiv.org/abs/2112.10752)
*   **Multimodal RAG**: [LlamaIndex Multimodal Guide](https://docs.llamaindex.ai/en/stable/examples/multimodal/MultimodalRAG/)

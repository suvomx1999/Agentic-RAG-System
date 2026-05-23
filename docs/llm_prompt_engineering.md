# Large Language Models and Prompt Engineering

## Large Language Models

### Architecture
Modern LLMs are based on the Transformer architecture:
- **Self-attention mechanism**: Enables the model to consider relationships between all tokens
- **Decoder-only**: Models like GPT, Gemini, and Llama generate text autoregressively
- **Scale**: Billions of parameters trained on trillions of tokens
- **Context window**: The maximum number of tokens the model can process at once

### Google Gemini
Google's Gemini family of models:
- **Gemini 2.5 Pro**: Most capable model, excellent for complex reasoning and analysis
- **Gemini 2.5 Flash**: Optimized for speed and efficiency, good balance of quality and cost
- **Gemini 2.5 Flash-Lite**: Fastest model, best for simple tasks and high throughput
- All models support text, image, audio, and video inputs (multimodal)
- Available through the Gemini API with generous free tier
- Supports function calling, structured output, and system instructions

### Temperature and Sampling
- **Temperature**: Controls randomness (0 = deterministic, 1 = creative)
- **Top-p (nucleus sampling)**: Only considers tokens within cumulative probability p
- **Top-k**: Only considers the k most likely next tokens
- For RAG applications, lower temperature (0-0.3) is recommended for factual accuracy

## Prompt Engineering

### System Prompts
System prompts set the behavior and constraints for the LLM:
- Define the model's role and expertise
- Set output format requirements (JSON, markdown, etc.)
- Establish guardrails and limitations
- Should be clear, specific, and concise

### Few-Shot Prompting
Providing examples in the prompt to guide model behavior:
- Include 2-5 input/output examples
- Examples should cover edge cases
- Consistent formatting across examples
- More examples generally improve consistency

### Chain-of-Thought (CoT)
Encouraging step-by-step reasoning:
- "Let's think step by step" improves complex reasoning
- Can be combined with few-shot examples
- Particularly effective for math and logic problems
- Self-consistency: generate multiple CoT chains and vote

### Structured Output
Requesting specific output formats:
- JSON output for programmatic parsing
- Markdown for human-readable responses
- Key: specify exact schema in system prompt
- Always include error handling for malformed output

## Tool Use and Function Calling

### Overview
Modern LLMs support function/tool calling:
- Model receives function definitions with parameter schemas
- Model generates structured function calls when appropriate
- Results are fed back to the model for response generation
- Enables LLMs to interact with external systems

### Best Practices for Tool Definitions
1. Clear, descriptive function names
2. Detailed parameter descriptions
3. Type annotations for all parameters
4. Examples in the description when helpful
5. Error handling for invalid tool calls

## Context Window Management

### Token Counting
- Different tokenizers for different models
- Tiktoken is commonly used for OpenAI models
- Context window includes: system prompt + user query + retrieved docs + generated response
- Leave headroom for generation (at least 1000 tokens)

### Context Strategies
- **Stuffing**: Put all relevant documents in the prompt (simple, limited by context window)
- **Map-reduce**: Process each document separately, then combine results
- **Refine**: Iteratively refine the answer with each document
- **Map-rerank**: Generate answers from each doc, rank and select the best

### Handling Long Documents
1. Use chunking with overlap to preserve context at boundaries
2. Apply summarization for very long documents
3. Use hierarchical retrieval (section → paragraph → sentence)
4. Consider models with larger context windows for document QA

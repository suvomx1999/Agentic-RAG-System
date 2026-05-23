# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by retrieving relevant information from external knowledge bases before generating responses. This approach combines the parametric knowledge stored in the LLM with non-parametric knowledge from a retrieval system.

## Core Components

### 1. Document Ingestion
The first step in any RAG system is ingesting documents into a searchable format. This involves:
- **Loading**: Reading documents from various formats (PDF, text, markdown, HTML)
- **Chunking**: Splitting documents into smaller, semantically meaningful segments
- **Embedding**: Converting text chunks into dense vector representations
- **Indexing**: Storing embeddings in a vector database for efficient similarity search

### 2. Retrieval
When a user query arrives, the retrieval component finds the most relevant document chunks:
- **Dense Retrieval**: Uses embedding similarity (cosine distance) to find semantically similar chunks
- **Sparse Retrieval (BM25)**: Uses term frequency-inverse document frequency for keyword matching
- **Hybrid Retrieval**: Combines dense and sparse methods using techniques like Reciprocal Rank Fusion (RRF)

### 3. Generation
The retrieved context is combined with the original query and sent to the LLM:
- The LLM generates an answer grounded in the retrieved documents
- Source citations are included to provide transparency
- The response quality depends heavily on the retrieval quality

## Advanced RAG Techniques

### Query Rewriting
Query rewriting transforms the original user query to improve retrieval:
- **HyDE (Hypothetical Document Embedding)**: Generate a hypothetical answer, then use it as the search query
- **Step-back Prompting**: Abstract the query to a broader concept for better background retrieval
- **Multi-query Expansion**: Generate multiple alternative phrasings to increase recall

### Reranking
After initial retrieval, a reranker model scores query-document pairs more accurately:
- Cross-encoder models like ms-marco-MiniLM evaluate query-document relevance jointly
- Reranking is computationally expensive but significantly improves precision
- Typically applied to top-K retrieved documents (K=20-50) and returns top-N (N=3-5)

### Agentic RAG
Agentic RAG adds decision-making capabilities to the RAG pipeline:
- An agent decides whether retrieval is necessary for simple queries
- The agent can choose between different retrieval strategies
- Self-reflection loops grade retrieved documents and generated answers
- Web search fallback is triggered when the knowledge base cannot answer the query
- Iterative refinement continues until the answer quality meets a threshold

## Evaluation Metrics

### RAGAS Framework
RAGAS (Retrieval-Augmented Generation Assessment) provides standardized metrics:

1. **Context Precision**: Measures what fraction of retrieved contexts are relevant to the query
2. **Context Recall**: Measures what fraction of the ground truth answer can be attributed to the retrieved context
3. **Faithfulness**: Evaluates whether the generated answer is grounded in the retrieved context (no hallucination)
4. **Answer Relevancy**: Checks if the generated answer actually addresses the user's question
5. **Answer Correctness**: Compares the generated answer against a ground truth reference answer
6. **Answer Similarity**: Measures semantic similarity between generated and reference answers

## Common Challenges

1. **Chunking Strategy**: Too large chunks dilute relevant information; too small chunks lose context
2. **Embedding Model Selection**: Different models perform better on different domains
3. **Hallucination**: LLMs may generate plausible but unsupported claims
4. **Multi-hop Reasoning**: Questions requiring information from multiple documents are harder
5. **Latency**: Multiple retrieval and generation steps increase response time
6. **Evaluation**: Ground truth creation for evaluation datasets is labor-intensive

## Best Practices

- Use hybrid retrieval (dense + sparse) for robust recall
- Apply reranking to improve precision after initial retrieval
- Implement answer grading to catch hallucinations
- Set maximum iteration limits to prevent infinite loops
- Cache intermediate results for efficiency
- Monitor retrieval quality metrics continuously in production

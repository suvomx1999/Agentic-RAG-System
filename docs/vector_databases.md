# Vector Databases and Embedding Models

## Vector Databases

### What are Vector Databases?
Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vector embeddings. Unlike traditional databases that search by exact matches or keyword patterns, vector databases perform similarity search using distance metrics like cosine similarity, Euclidean distance, or dot product.

### ChromaDB
ChromaDB is an open-source, lightweight vector database designed for AI applications:
- **Persistent storage**: Data survives application restarts with PersistentClient
- **Collections**: Logical groupings of embeddings, similar to tables in relational databases
- **Metadata filtering**: Query results can be filtered by metadata fields
- **Built-in embedding**: Can automatically embed documents using default or custom embedding functions
- **API simplicity**: Simple add/query/delete operations

### Key Operations
1. **Upsert**: Add or update documents and their embeddings
2. **Query**: Find the most similar documents to a query embedding
3. **Filter**: Combine similarity search with metadata conditions
4. **Delete**: Remove documents by ID or metadata filter

### Indexing Algorithms
Vector databases use approximate nearest neighbor (ANN) algorithms:
- **HNSW (Hierarchical Navigable Small World)**: Graph-based, excellent recall-speed tradeoff
- **IVF (Inverted File Index)**: Clustering-based, good for large datasets
- **PQ (Product Quantization)**: Compression technique to reduce memory usage
- **ScaNN**: Google's scalable nearest neighbor search

## Embedding Models

### What are Embeddings?
Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search even when queries don't share keywords with relevant documents.

### Popular Models

#### BAAI/bge-m3
- Multilingual model supporting 100+ languages
- Supports dense retrieval, sparse retrieval, and multi-vector retrieval
- 1024-dimensional embeddings
- Context length: 8192 tokens
- Open source and free to use
- Excellent performance on MTEB benchmarks

#### OpenAI text-embedding-3-small
- 1536-dimensional embeddings (configurable via dimensions parameter)
- Cost-effective API-based embedding
- Good performance for English text
- Supports dimension reduction for faster search

#### Sentence-BERT Models
- all-MiniLM-L6-v2: Fast, lightweight (384 dimensions)
- all-mpnet-base-v2: Higher quality (768 dimensions)
- Designed specifically for semantic similarity tasks

### Embedding Best Practices
1. Choose embedding dimension based on accuracy vs. speed tradeoff
2. Normalize embeddings for cosine similarity calculations
3. Use the same model for both indexing and querying
4. Consider domain-specific fine-tuning for specialized applications
5. Batch embedding operations for efficiency

## BM25 (Best Matching 25)

### Overview
BM25 is a probabilistic information retrieval algorithm that ranks documents based on term frequency and document length:
- Does not require neural networks or GPU
- Excellent at exact keyword matching
- Complementary to dense retrieval for hybrid search

### Algorithm
BM25 scores are calculated using:
- Term Frequency (TF): How often a term appears in a document
- Inverse Document Frequency (IDF): How rare a term is across the corpus
- Document length normalization: Adjusts for varying document sizes
- Parameters k1 and b control saturation and length normalization

### BM25Okapi
The most common BM25 variant used in practice:
- Default parameters: k1=1.5, b=0.75
- Fast computation, no training required
- Can be serialized with pickle for persistence
- Works well with simple tokenization (whitespace + lowercase)

## Hybrid Retrieval

### Reciprocal Rank Fusion (RRF)
RRF combines results from multiple retrieval systems:
- For each document, compute: RRF_score = Σ(1 / (k + rank_i))
- k is a constant (typically 60) that prevents high-ranked items from dominating
- Results from all systems are merged and sorted by RRF score
- Deduplication ensures each document appears only once

### Benefits of Hybrid Retrieval
- Dense retrieval captures semantic similarity
- Sparse retrieval captures keyword matches
- Combined approach has higher recall than either method alone
- RRF is robust and doesn't require calibration between score scales

## Cross-Encoder Reranking

### How Reranking Works
Cross-encoder rerankers score query-document pairs jointly:
1. Initial retrieval returns a broad set of candidates (e.g., top 50)
2. Reranker scores each (query, document) pair
3. Results are re-sorted by reranker scores
4. Top-N documents are selected for generation

### ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking dataset
- 22M parameters, fast inference on CPU
- Input: concatenated query and document
- Output: relevance score (higher = more relevant)
- Maximum sequence length: 512 tokens
- Excellent balance of speed and accuracy for English queries

### When to Use Reranking
- Always beneficial when initial retrieval returns more than needed
- Particularly valuable for ambiguous queries
- Computational cost is O(N) where N is the number of candidates
- Not needed when retrieval returns fewer than top-N documents

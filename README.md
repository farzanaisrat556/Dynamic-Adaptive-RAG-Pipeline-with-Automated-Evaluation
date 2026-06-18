
# Dynamic Adaptive RAG Pipeline with Automated Evaluation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-red)

A research-oriented implementation of an **Adaptive Retrieval-Augmented Generation (RAG)** framework that dynamically routes user queries based on estimated reasoning complexity. The project benchmarks a conventional single-shot RAG pipeline against an adaptive retrieval strategy and evaluates performance using automated NLP metrics on open-domain question-answering datasets.

---

## Overview

Traditional RAG systems typically perform a single retrieval step before generation. While effective for straightforward factual queries, this approach often struggles with:

* Multi-hop reasoning
* Comparative questions
* Ambiguous information needs
* Context fragmentation across documents

This project introduces an **Adaptive RAG Pipeline** that selectively applies additional retrieval and query refinement operations when a query is predicted to require deeper reasoning.

The objective is to investigate whether adaptive retrieval strategies can improve answer quality while maintaining retrieval efficiency.

---

## Key Features

### Adaptive Query Routing

Automatically classifies incoming questions and routes them through:

* Standard retrieval path
* Multi-step retrieval path
* Query rewriting path

depending on estimated query complexity.

### Dense Semantic Retrieval

Uses transformer-based sentence embeddings and FAISS indexing for efficient similarity search.

### Baseline vs Adaptive Benchmarking

Provides a controlled experimental framework for comparing:

* Retrieval quality
* Generation quality
* Semantic alignment

between conventional and adaptive RAG architectures.

### Automated Evaluation Engine

Computes:

* ROUGE-L
* BERTScore
* Comparative performance deltas

against ground-truth answers.

### Reproducible Research Workflow

Supports benchmarking on public QA datasets including:

* HotpotQA
* RetrievalQA
* Custom document corpora

---

# System Architecture

```text
                     ┌──────────────────┐
                     │ Raw Documents    │
                     └─────────┬────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Chunking Module  │
                     └─────────┬────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ Embedding Model  │
                     │ MiniLM-L6-v2     │
                     └─────────┬────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ FAISS Vector DB  │
                     └─────────┬────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼

        Baseline RAG                 Adaptive Router
              │                            │
              ▼                            ▼
      Top-K Retrieval            Query Analysis
              │                  Query Rewriting
              │                  Multi-Hop Retrieval
              ▼                            │
       Llama-3 Generator  ◄───────────────┘
              │
              ▼
        Generated Answer
              │
              ▼
      Evaluation Pipeline
    (ROUGE-L + BERTScore)
```

---

# Methodology

## 1. Document Processing

Documents are segmented using LangChain's Recursive Character Text Splitter to create semantically coherent chunks.

## 2. Embedding Generation

Each chunk is embedded using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are indexed into a local FAISS vector store.

## 3. Baseline Retrieval-Augmented Generation

The baseline system:

1. Retrieves Top-K passages
2. Constructs a prompt
3. Generates an answer using Llama-3

```text
Question
   ↓
Retrieve Top-K Contexts
   ↓
Generate Answer
```

## 4. Adaptive Retrieval-Augmented Generation

The adaptive system introduces an intermediate routing layer.

For complex queries:

1. Analyze query complexity
2. Rewrite the query
3. Perform secondary retrieval
4. Aggregate retrieved evidence
5. Generate final answer

```text
Question
   ↓
Complexity Assessment
   ↓
Adaptive Retrieval
   ↓
Evidence Aggregation
   ↓
Generation
```

---

# Experimental Setup

| Component          | Configuration            |
| ------------------ | ------------------------ |
| Embedding Model    | all-MiniLM-L6-v2         |
| Vector Database    | FAISS                    |
| Generator          | Meta-Llama-3-8B-Instruct |
| Evaluation Metrics | ROUGE-L, BERTScore       |
| Dataset            | HotpotQA                 |
| Framework          | LangChain                |

---

# Results

## Sample Benchmark

| Query Type             | Baseline ROUGE-L | Adaptive ROUGE-L |
| ---------------------- | ---------------- | ---------------- |
| Multi-hop Question A   | 0.2000           | **0.2105**       |
| Comparative Question B | 0.1200           | **0.2105**       |

The adaptive routing strategy demonstrates measurable improvements on selected complex reasoning tasks where single-pass retrieval fails to capture sufficient supporting evidence.

---

# Repository Structure

```text
adaptive-rag/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── embeddings/
│
├── vectorstore/
│
├── src/
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── adaptive_router.py
│   ├── generation.py
│   └── evaluation.py
│
├── notebooks/
│
├── results/
│   ├── benchmark_results.csv
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/your-username/adaptive-rag.git

cd adaptive-rag

pip install -r requirements.txt
```

---

# Requirements

```bash
faiss-cpu
sentence-transformers
transformers
accelerate
langchain
langchain-huggingface
datasets
rouge-score
bert-score
pandas
numpy
tqdm
```

---

# Future Work

### LLM-Based Routing

Replace heuristic complexity estimation with an LLM-driven router.

### RAGAS Integration

Evaluate:

* Faithfulness
* Context Precision
* Context Recall
* Answer Relevance

using the RAGAS framework.

### Hybrid Retrieval

Combine:

* Dense retrieval
* Sparse retrieval (BM25)
* Reciprocal Rank Fusion

for improved evidence coverage.

### Asynchronous Retrieval

Introduce parallel retrieval and generation pipelines to reduce latency in production-scale deployments.

---

# Research Contribution

This project provides a reproducible experimental framework for studying adaptive retrieval strategies in Retrieval-Augmented Generation systems. The implementation demonstrates how query-aware retrieval routing can improve performance on multi-hop and reasoning-intensive question-answering tasks while maintaining compatibility with standard vector-based RAG architectures.

---

# References

1. Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
2. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
3. LangChain Documentation.
4. Hugging Face Transformers.
5. FAISS Similarity Search Library.

---
title: Reddit Sentiment & Insight Analyzer
emoji: 📊
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Reddit Sentiment & Insight Analyzer - Advanced RAG Edition

## 1. Problem Statement
In today's competitive digital market, user feedback is gold. However, a significant portion of honest, unfiltered feedback is scattered across Reddit—buried in unstructured text, complaint threads, and rants. Manually sifting through thousands of Reddit posts to identify common bugs, missing features, and overall sentiment for a specific product is incredibly time-consuming and prone to human bias. 

This project provides an automated, enterprise-grade AI system that instantly scrapes unstructured Reddit feedback, understands the context via a state-of-the-art Hybrid RAG pipeline, and extracts actionable insights (top bugs, hated features, emotional intensity) without requiring manual review.

## 2. Complete Workflow
The project implements a cutting-edge Retrieval-Augmented Generation (RAG) pipeline combined with headless browser scraping and asynchronous AI summarization.

### Phase 1: Data Acquisition (Apify Scraping)
*   **User Input:** The user enters a product name (e.g., "Instagram", "Dell laptop") in the Glassmorphism Web UI.
*   **Headless Scraping:** The `embedder.py` script utilizes the `Apify` API (`automation-lab/reddit-scraper`) to bypass Reddit's strict API rate limits and 403 blocks. It scrapes top posts and comments related to the product query.
*   **Deduplication:** Posts are deterministically hashed (MD5) based on URL and content to prevent database bloat before vectorization.

### Phase 2: Hybrid Data Processing & Embedding
*   **Chunking:** LangChain's `RecursiveCharacterTextSplitter` chunks the text into manageable pieces to preserve context.
*   **Dense Vectors:** Chunks are transformed into high-dimensional semantic vectors using the `sentence-transformers/all-MiniLM-L6-v2` HuggingFace model.
*   **Sparse Vectors (BM25):** The system dynamically trains a `BM25Encoder` (via `pinecone-text`) on the scraped corpus to generate exact-keyword sparse vectors, saving the model locally.
*   **Storage:** The Hybrid Embeddings (Dense + Sparse) are uploaded to a Serverless **Pinecone** vector database using the `dotproduct` metric.

### Phase 3: Advanced Semantic Retrieval (RAG Pipeline)
When analyzing complaints or answering chat questions, `rag_pipeline.py` executes a multi-stage retrieval process:
1.  **Multi-Query Expansion:** An LLM (`openai/gpt-oss-120b` via Groq) takes the user's query and generates 3 alternative variations to ensure no edge-case complaints are missed.
2.  **Hybrid Search:** The system queries Pinecone using both semantic meaning (Dense) and exact keyword matches (Sparse) for all 3 generated queries, pulling a massive net of 100 relevant documents.
3.  **Cross-Encoder Re-Ranking:** A `CustomCrossEncoderReranker` runs the 100 documents through a heavy-duty `cross-encoder/ms-marco-MiniLM-L-6-v2` neural network to strictly re-score and filter them down to the true Top 40 most relevant documents.

### Phase 4: AI Analysis & Map-Reduce
*   **Map-Reduce Chain:** Because the Top 40 documents exceed standard LLM token limits, `analyzer.py` utilizes a Map-Reduce summarization chain to process the chunks in batches without crashing.
*   **Structured Output:** The Groq LLM (`openai/gpt-oss-120b`) reads the context and returns a strictly formatted JSON response containing the most hated features, common bugs, sentiment, emotional intensity, and a summary.
31. 
32. ### Phase 5: Interactive Visualization (UI)
33. *   **Dashboard:** The frontend (`index.html`/`script.js`), built with Vanilla JS, CSS, and HTML, features a stunning, animated **Glassmorphism** dark-mode UI.
34. *   **Visuals:** It dynamically renders custom charts (using Chart.js), metrics, and deep-dive lists of bugs.
35: *   **Conversational AI:** Provides an interactive chat interface where users can ask follow-up questions to the LLM, strictly grounded in the scraped Reddit data context.
36: 
37: ## 3. Architecture & Tech Stack
38: 
39: *   **Frontend:** HTML5, CSS3 (Glassmorphism, animations), Vanilla JavaScript, Chart.js.
40: *   **Backend:** FastAPI (Asynchronous, non-blocking I/O via Uvicorn).
41: *   **Data Ingestion:** Apify Client (`automation-lab/reddit-scraper`).
42: *   **Vector Database:** Pinecone (Serverless, dotproduct metric for Hybrid Search).
43: *   **Embedding Models:** HuggingFace `all-MiniLM-L6-v2` (Dense) & `pinecone-text` BM25 (Sparse).
44: *   **Reranking Model:** HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2`.
45: *   **LLM / Inference:** Groq API (`openai/gpt-oss-120b` for routing, expansion, and structured analysis).
46: *   **Orchestration:** LangChain (MultiQueryRetriever, ContextualCompressionRetriever, Map-Reduce Chains).

## 4. Local Setup & Installation

### Prerequisites
1. Python 3.10+
2. API Keys for: **Groq**, **Pinecone**, and **Apify**.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sivarajs24/Reddit_rage_analyzer.git
   cd Reddit_rage_analyzer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Environment Variables:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=your_index_name (MUST be created with 'dotproduct' metric)
   APIFY_API_TOKEN=your_apify_key
   ```
5. Run the Application locally:
   ```bash
   uvicorn api:app --reload
   ```
6. Open your browser and navigate to `http://127.0.0.1:8000`.

## 5. Deployment Options

### Docker Deployment
Build and run the container locally or on any cloud container service (e.g. AWS ECS, GCP Cloud Run, Render):
```bash
docker build -t reddit-sentiment-analyzer .
docker run -p 8000:8000 --env-file .env reddit-sentiment-analyzer
```

### PaaS Platforms (Render / Railway / Heroku)
The project includes a `Procfile` and dynamic `$PORT` binding.
1. Connect your repository to Render, Railway, or Heroku.
2. Configure environment variables (`GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `APIFY_API_TOKEN`).
3. Deploy as a Web Service. The service will automatically launch with `uvicorn api:app --host 0.0.0.0 --port $PORT`.

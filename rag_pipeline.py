import os
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from typing import Sequence
from langchain_groq import ChatGroq
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import logging
from pydantic import Field
from embedder import get_embedding_model

# Set logging for MultiQueryRetriever to see the generated queries
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    model_name: str = CROSS_ENCODER_MODEL_NAME
    top_n: int = 40
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
            
        print(f"Reranking {len(documents)} documents down to {self.top_n}...")
        try:
            import gc
            import torch
            from sentence_transformers import CrossEncoder
            with torch.no_grad():
                model = CrossEncoder(self.model_name, device='cpu')
                pairs = [[query, doc.page_content] for doc in documents]
                scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
                del model
                gc.collect()
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.top_n]]
        except Exception as e:
            print(f"Note: CrossEncoder re-ranking skipped ({e}). Using top {self.top_n} hybrid search results.")
            return list(documents)[:self.top_n]

def get_advanced_retriever(product_name: str, k: int = 40):
    collection_name = f"reddit_{product_name.replace(' ', '_').lower()}"
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    if not index_name:
        return None
        
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # 1. Load BM25 Sparse Encoder for Hybrid Search
    bm25_path = f"bm25_models/bm25_{collection_name}.json"
    bm25_encoder = BM25Encoder().default()
    if os.path.exists(bm25_path):
        bm25_encoder.load(bm25_path)
    else:
        print("Warning: BM25 encoder not found for this product. Hybrid search might be degraded.")
        
    embeddings = get_embedding_model()
    
    # 2. Base Hybrid Retriever (Retrieves top 100 for re-ranking)
    hybrid_retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
        namespace=collection_name,
        top_k=100
    )
    
    # 3. Multi-Query Expansion
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(temperature=0, model_name=model_name)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )
    
    # 4. Cross-Encoder Re-Ranking
    reranker = CustomCrossEncoderReranker(top_n=k)
    
    # 5. Final Compression Retriever
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=multi_query_retriever
    )
    
    return advanced_retriever

def retrieve_complaints(query: str, product_name: str, k: int = 40):
    retriever = get_advanced_retriever(product_name, k)
    if not retriever:
        return []
        
    print(f"Retrieving top {k} complaints using Advanced RAG for query: '{query}'")
    docs = retriever.invoke(query)
    return docs

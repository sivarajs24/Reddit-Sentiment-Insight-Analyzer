import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from typing import Sequence
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import logging
from pydantic import Field

# Set logging for MultiQueryRetriever to see the generated queries
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    model_name: str = CROSS_ENCODER_MODEL_NAME
    top_n: int = 40
    
    # We must use PrivateAttr to prevent pydantic from trying to parse the CrossEncoder object
    # But for simplicity, we just initialize it inside compress_documents on first run, 
    # or use a property.
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
            
        print(f"Reranking {len(documents)} documents down to {self.top_n}...")
        model = CrossEncoder(self.model_name)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:self.top_n]]

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
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
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

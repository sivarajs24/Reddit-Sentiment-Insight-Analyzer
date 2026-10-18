import os
import warnings

warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from scraper import scrape_reddit
from embedder import process_product
from rag_pipeline import retrieve_complaints
from analyzer import analyze_complaints
from groq import Groq

load_dotenv()

app = FastAPI(title="Reddit Sentiment & Insight Analyzer API")

# Enable CORS for deployment across different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class AnalyzeRequest(BaseModel):
    product_name: str
    custom_subreddits: Optional[str] = ""
    focus: Optional[str] = ""

class ChatRequest(BaseModel):
    product_name: str
    prompt: str

@app.get("/")
async def root():
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Reddit Sentiment & Insight Analyzer",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "pinecone_configured": bool(os.getenv("PINECONE_API_KEY")),
        "apify_configured": bool(os.getenv("APIFY_API_TOKEN")),
    }

@app.get("/{filename}")
async def serve_file(filename: str):
    file_path = f"frontend/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse('frontend/index.html')

@app.post("/api/analyze")
async def api_analyze(request: AnalyzeRequest):
    product_name = request.product_name.strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="Product name required")

    custom_subs = [s.strip() for s in request.custom_subreddits.split(",") if s.strip()]
    
    # Use the dynamic focus if provided, otherwise default to the classic "Rage" theme query
    query = request.focus.strip() if request.focus and request.focus.strip() else "major issues bugs hate worst features"

    try:
        # Run blocking synchronous functions in a threadpool so they don't block the FastAPI event loop
        raw_data = await run_in_threadpool(scrape_reddit, product_name, 10, custom_subs)
        if not raw_data:
            raise HTTPException(status_code=404, detail="No data found or scraping failed.")

        success = await run_in_threadpool(process_product, raw_data, product_name)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process embeddings.")

        retrieved_docs = await run_in_threadpool(retrieve_complaints, query, product_name, 40)
        if not retrieved_docs:
            raise HTTPException(status_code=500, detail="RAG retrieval failed.")

        analysis_result = await analyze_complaints(product_name, retrieved_docs)
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {analysis_result['error']}")

        # Format docs for JSON serialization
        docs_json = []
        for doc in retrieved_docs:
            docs_json.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        return {
            "analysis_result": analysis_result,
            "retrieved_docs": docs_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    product_name = request.product_name
    prompt = request.prompt

    try:
        new_docs = await run_in_threadpool(retrieve_complaints, prompt, product_name, 10)
        context = ""
        for i, doc in enumerate(new_docs):
            context += f"--- Document {i+1} ---\n{doc.page_content}\n\n"
            
        system_prompt = f"You are a helpful AI assistant answering questions about {product_name} complaints. Base your answers strictly on the context provided below.\n\nContext:\n{context}"
        
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.2,
        )
        response = chat_completion.choices[0].message.content
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

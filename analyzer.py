import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

class AnalysisResult(BaseModel):
    most_hated_features: List[str] = Field(description="List of the most hated features (or best features if positive)")
    common_bug_patterns: List[str] = Field(description="List of common bug patterns or issues")
    emotional_intensity: str = Field(description="Emotional intensity: High/Medium/Low")
    overall_sentiment: str = Field(description="Overall sentiment: Positive/Neutral/Negative/Very Negative")
    key_recurring_complaints: List[str] = Field(description="List of key recurring feedback points")
    summary_paragraph: str = Field(description="A professional 3-4 sentence summary of the general consensus")
    top_keywords: List[str] = Field(description="List of top keywords associated with the feedback")

async def _summarize_batch(client: AsyncGroq, batch_docs: list, product_name: str, batch_index: int) -> str:
    """Map phase: Summarize a small batch of documents."""
    context = ""
    for i, doc in enumerate(batch_docs):
        context += f"--- Thread {i+1} ---\n{doc.page_content}\n\n"
        
    prompt = f"""
    You are an expert product analyst. Extract the key insights, sentiments, bugs, and opinions from this batch of Reddit threads about '{product_name}'.
    Keep it concise but detailed.
    
    --- DATA ---
    {context}
    """
    
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.2,
        )
        return f"--- BATCH {batch_index} SUMMARY ---\n{response.choices[0].message.content}\n\n"
    except Exception as e:
        print(f"Error in batch {batch_index}: {e}")
        return ""

async def analyze_complaints(product_name: str, retrieved_docs: list) -> dict:
    """
    Map-Reduce RAG analysis to prevent Token Limit Exceeded errors.
    """
    if not retrieved_docs:
        return {"error": "No documents provided for analysis."}
        
    if not os.getenv("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY is not set in .env."}

    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    
    # 1. MAP PHASE: Split documents into batches of 10
    batch_size = 10
    batches = [retrieved_docs[i:i + batch_size] for i in range(0, len(retrieved_docs), batch_size)]
    
    print(f"Map Phase: Splitting {len(retrieved_docs)} docs into {len(batches)} batches for parallel processing.")
    
    # Run summaries in parallel
    tasks = [
        _summarize_batch(client, batch, product_name, idx + 1)
        for idx, batch in enumerate(batches)
    ]
    batch_summaries = await asyncio.gather(*tasks)
    
    # Combine the reduced summaries
    final_context = "".join(batch_summaries)
    
    # 2. REDUCE PHASE: Final structured JSON output
    print("Reduce Phase: Generating final structured JSON analysis.")
    prompt = f"""
    You are an expert AI product analyst. I have processed thousands of Reddit threads about '{product_name}' and summarized them into batches below.
    
    Based ONLY on the batch summaries provided below, generate a comprehensive final analysis and return the result as a strictly formatted JSON object. 
    The JSON structure MUST exactly match the following schema:
    {AnalysisResult.model_json_schema()}
    
    --- BATCH SUMMARIES ---
    {final_context}
    """

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that only outputs strictly valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result_text = chat_completion.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Validate against Pydantic model
        validated_result = AnalysisResult(**result_json)
        return validated_result.model_dump()

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from Groq response: {e}")
        return {"error": "Failed to parse AI response into JSON format."}
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"error": str(e)}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import requests
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = "You are a helpful assistant."
    model: Optional[str] = "gpt-3.5-turbo"
    api_key: str
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/models")
async def get_models():
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        models_data = response.json()
        
        # Filter for text-to-text models only
        text_models = []
        for model in models_data.get("data", []):
            modalities = model.get("architecture", {})
            input_modalities = modalities.get("input_modalities", [])
            output_modalities = modalities.get("output_modalities", [])
            
            # Only include models that support text input and text output
            if "text" in input_modalities and "text" in output_modalities:
                model_id = model.get("id", "")
                model_name = model.get("name", model_id)
                context_length = model.get("context_length", 0)
                
                # Extract model size from ID or name
                size_info = "Unknown"
                size_patterns = ["405b", "70b", "72b", "34b", "22b", "13b", "8b", "7b", "3b", "1b", "500m", "100m"]
                for size in size_patterns:
                    if size in model_id.lower() or size in model_name.lower():
                        size_info = size.upper()
                        break
                
                # Format context length
                if context_length >= 1000000:
                    context_display = f"{context_length//1000000}M"
                elif context_length >= 1000:
                    context_display = f"{context_length//1000}K"
                else:
                    context_display = str(context_length) if context_length > 0 else "Unknown"
                
                # Get pricing
                pricing = model.get("pricing", {})
                prompt_price = pricing.get("prompt", "0")
                completion_price = pricing.get("completion", "0")
                
                try:
                    prompt_cost = float(prompt_price) * 1000000 if prompt_price else 0
                    completion_cost = float(completion_price) * 1000000 if completion_price else 0
                    if prompt_cost == 0 and completion_cost == 0:
                        price_display = "FREE"
                    else:
                        price_display = f"${prompt_cost:.2f}/${completion_cost:.2f}"
                except:
                    price_display = "Unknown"
                
                text_models.append({
                    "id": model_id,
                    "name": model_name,
                    "size": size_info,
                    "context": context_display,
                    "context_length": context_length,
                    "price": price_display,
                    "prompt_cost": float(prompt_price) * 1000000 if prompt_price else 0,
                    "completion_cost": float(completion_price) * 1000000 if completion_price else 0,
                    "description": model.get("description", "")
                })
        
        # Sort by prompt cost (cheapest first)
        text_models.sort(key=lambda x: x["prompt_cost"])
        
        return {"models": text_models}
        
    except Exception as e:
        # Minimal fallback
        return {"models": [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "size": "Unknown", "context": "16K", "context_length": 16000, "price": "$0.50/$1.50", "prompt_cost": 0.5, "completion_cost": 1.5, "description": "Fast and capable"}
        ]}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Build messages for OpenRouter API
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Call OpenRouter API directly
        headers = {
            "Authorization": f"Bearer {request.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Chat App"
        }
        
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return ChatResponse(response=result["choices"][0]["message"]["content"])
        else:
            error_data = response.json() if response.content else {"error": "Unknown error"}
            return ChatResponse(response=f"Error: {error_data.get('error', {}).get('message', 'API request failed')}")
    
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
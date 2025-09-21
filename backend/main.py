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
        
        # Include ALL text-to-text models (exclude only vision, image generation, embedding, etc.)
        text_models = []
        for model in models_data.get("data", []):
            model_id = model.get("id", "").lower()
            model_name = model.get("name", "")
            
            # Exclude specific non-text models
            excluded_types = [
                "vision", "image", "embedding", "tts", "whisper", "dall-e", 
                "midjourney", "stable-diffusion", "flux", "playground-v", 
                "cogvideox", "luma", "kling", "runway", "ideogram"
            ]
            
            # Check if it's a text-to-text model (exclude vision/image/audio models)
            is_excluded = any(excluded in model_id for excluded in excluded_types)
            is_excluded = is_excluded or any(excluded in model_name.lower() for excluded in excluded_types)
            
            if not is_excluded:
                text_models.append({
                    "id": model.get("id"),
                    "name": model.get("name", model.get("id")),
                    "description": model.get("description", ""),
                    "pricing": model.get("pricing", {}),
                    "context_length": model.get("context_length", 0)
                })
        
        # Sort by name for better organization
        text_models.sort(key=lambda x: x["name"])
        
        return {"models": text_models}
    except Exception as e:
        # Fallback models if API fails
        return {"models": [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and capable"},
            {"id": "openai/gpt-4", "name": "GPT-4", "description": "Most capable model"},
            {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Latest GPT-4 model"},
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "description": "Most capable Claude model"},
            {"id": "anthropic/claude-3-sonnet", "name": "Claude 3 Sonnet", "description": "Balanced Claude model"},
            {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "description": "Fast Claude model"},
            {"id": "google/gemini-pro", "name": "Gemini Pro", "description": "Google's flagship model"},
            {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5", "description": "Latest Gemini model"},
            {"id": "x-ai/grok-beta", "name": "Grok Beta", "description": "xAI's Grok model"},
            {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B", "description": "Meta's Llama model"},
            {"id": "mistralai/mixtral-8x7b-instruct", "name": "Mixtral 8x7B", "description": "Mistral's mixture of experts"}
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
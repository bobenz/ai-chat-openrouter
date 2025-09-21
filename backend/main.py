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
    # Comprehensive list of text-to-text models from OpenRouter with sizes, context, and pricing
    return {"models": [
        # Free models
        {"id": "microsoft/wizardlm-2-8x22b", "name": "WizardLM 2 8x22B", "display_name": "WizardLM 2 [8x22B, 65K ctx, FREE]"},
        {"id": "google/gemma-2-9b-it:free", "name": "Gemma 2 9B (free)", "display_name": "Gemma 2 [9B, 8K ctx, FREE]"},
        {"id": "meta-llama/llama-3.1-8b-instruct:free", "name": "Llama 3.1 8B (free)", "display_name": "Llama 3.1 [8B, 128K ctx, FREE]"},
        {"id": "microsoft/phi-3-mini-128k-instruct:free", "name": "Phi-3 Mini (free)", "display_name": "Phi-3 Mini [3.8B, 128K ctx, FREE]"},
        
        # Ultra-cheap models
        {"id": "meta-llama/llama-3.2-1b-instruct", "name": "Llama 3.2 1B", "display_name": "Llama 3.2 [1B, 128K ctx, $0.04/$0.04/1M]"},
        {"id": "meta-llama/llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "display_name": "Llama 3.2 [3B, 128K ctx, $0.06/$0.06/1M]"},
        {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B", "display_name": "Gemma 2 [9B, 8K ctx, $0.08/$0.08/1M]"},
        {"id": "qwen/qwen-2-7b-instruct", "name": "Qwen 2 7B", "display_name": "Qwen 2 [7B, 128K ctx, $0.09/$0.09/1M]"},
        
        # Budget models
        {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B", "display_name": "Llama 3.1 [8B, 128K ctx, $0.18/$0.18/1M]"},
        {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B", "display_name": "Mistral [7B, 32K ctx, $0.20/$0.20/1M]"},
        {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "display_name": "Claude 3 Haiku [200K ctx, $0.25/$1.25/1M]"},
        {"id": "google/gemini-flash-1.5", "name": "Gemini 1.5 Flash", "display_name": "Gemini 1.5 Flash [1M ctx, $0.38/$1.13/1M]"},
        
        # Mid-tier models
        {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "display_name": "GPT-3.5 Turbo [16K ctx, $0.50/$1.50/1M]"},
        {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B", "display_name": "Llama 3.1 [70B, 128K ctx, $0.88/$0.88/1M]"},
        {"id": "mistralai/mixtral-8x7b-instruct", "name": "Mixtral 8x7B", "display_name": "Mixtral [8x7B, 32K ctx, $0.24/$0.24/1M]"},
        {"id": "mistralai/mixtral-8x22b-instruct", "name": "Mixtral 8x22B", "display_name": "Mixtral [8x22B, 65K ctx, $1.20/$1.20/1M]"},
        {"id": "google/gemini-pro-1.5", "name": "Gemini 1.5 Pro", "display_name": "Gemini 1.5 Pro [2M ctx, $1.25/$5.00/1M]"},
        
        # Premium models
        {"id": "anthropic/claude-3-sonnet", "name": "Claude 3 Sonnet", "display_name": "Claude 3 Sonnet [200K ctx, $3.00/$15.00/1M]"},
        {"id": "cohere/command-r", "name": "Command R", "display_name": "Command R [128K ctx, $0.50/$1.50/1M]"},
        {"id": "cohere/command-r-plus", "name": "Command R+", "display_name": "Command R+ [128K ctx, $3.00/$15.00/1M]"},
        {"id": "x-ai/grok-beta", "name": "Grok Beta", "display_name": "Grok Beta [131K ctx, $5.00/$15.00/1M]"},
        
        # High-end models
        {"id": "openai/gpt-4o", "name": "GPT-4o", "display_name": "GPT-4o [128K ctx, $5.00/$15.00/1M]"},
        {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo", "display_name": "GPT-4 Turbo [128K ctx, $10.00/$30.00/1M]"},
        {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "display_name": "Claude 3 Opus [200K ctx, $15.00/$75.00/1M]"},
        {"id": "openai/gpt-4", "name": "GPT-4", "display_name": "GPT-4 [8K ctx, $30.00/$60.00/1M]"},
        
        # Specialized models
        {"id": "perplexity/llama-3.1-sonar-large-128k-online", "name": "Llama 3.1 Sonar 70B Online", "display_name": "Llama 3.1 Sonar [70B, 128K ctx, Online, $1.00/$1.00/1M]"},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat", "display_name": "DeepSeek Chat [67B, 32K ctx, $0.14/$0.28/1M]"},
        {"id": "meta-llama/llama-3.1-405b-instruct", "name": "Llama 3.1 405B", "display_name": "Llama 3.1 [405B, 128K ctx, $5.32/$16.00/1M]"},
        {"id": "mistralai/mistral-large", "name": "Mistral Large", "display_name": "Mistral Large [128K ctx, $2.00/$6.00/1M]"},
        {"id": "01-ai/yi-large", "name": "Yi Large", "display_name": "Yi Large [32K ctx, $3.00/$3.00/1M]"},
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
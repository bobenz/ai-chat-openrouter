# AI Chat Application

A simple AI chat application with a React-like frontend and FastAPI backend using LangChain.

## Features

- Clean, modern chat interface
- Configurable LLM settings (system prompt, model, API key, temperature)
- Real-time chat with OpenAI models
- Persistent settings storage
- CORS-enabled backend

## Setup

### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Open `index.html` in your browser or serve it with a simple HTTP server:
```bash
python -m http.server 3000
```

## Usage

1. Start the backend server
2. Open the frontend in your browser
3. Click "Settings" to configure:
   - System Prompt
   - Model (GPT-3.5, GPT-4, etc.)
   - OpenAI API Key
   - Temperature (0-2)
4. Start chatting!

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Send chat messages

## Technologies

- **Backend**: FastAPI, LangChain, OpenAI
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Styling**: Modern CSS with gradients and animations
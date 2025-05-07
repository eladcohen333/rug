# Hebrew RAG Chat Application

A Retrieval-Augmented Generation (RAG) chat application that uses local vector storage and Google Gemini for responses, optimized for Hebrew text processing.

## Project Structure

```
.
├── backend/           # FastAPI backend
├── frontend/         # React frontend
└── README.md
```

## Features

- Local RAG implementation using LangChain
- Vector storage with ChromaDB
- Modern React frontend with chat interface
- Integration with Google Gemini for responses

## Setup Instructions

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   uvicorn main:app --reload --port 5000
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```

## Environment Variables

Create a `.env` file in the backend directory with:
```
GOOGLE_API_KEY=your_api_key_here
``` 
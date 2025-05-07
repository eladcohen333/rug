from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini with API key
genai.configure(api_key="AIzaSyD4fbqXqNmstdusDPrZ88vnYTXhu8GzxeA")
model = genai.GenerativeModel('gemini-pro')

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
vector_store = Chroma(
    persist_directory="./data/chroma",
    embedding_function=embeddings
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Get the last user message
        user_message = request.messages[-1].content
        
        # Search for relevant documents
        docs = vector_store.similarity_search(user_message, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prepare prompt with context
        prompt = f"""Context: {context}
        
        User question: {user_message}
        
        Please provide a helpful response based on the context above."""
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file_path: str):
    try:
        # Load and process the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        vector_store.add_documents(chunks)
        vector_store.persist()
        
        return {"message": "Document processed and added to vector store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
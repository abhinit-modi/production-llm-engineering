# RAG Application

Full-stack Retrieval-Augmented Generation chatbot — FastAPI backend with vector search, conversation history, and Streamlit frontend.

## Overview

A production-ready chat application with three modes:

1. **Simple Chat** — Direct LLM responses
2. **History Chat** — Conversation memory with database persistence
3. **RAG Chat** — Retrieval-augmented responses with vector search

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    Backend      │────▶│   LLM Endpoint  │
│   (Streamlit)   │     │   (FastAPI)     │     │  (HuggingFace)  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ SQLite   │ │ Pinecone │ │ ChromaDB │
              │ (History)│ │ (Vectors)│ │ (Sources)│
              └──────────┘ └──────────┘ └──────────┘
```

## Features

### Backend (FastAPI)

- **Streaming Endpoints** — Server-Sent Events for real-time responses
- **Chat History** — SQLAlchemy + SQLite persistence per user
- **Vector Search** — Pinecone for document retrieval
- **Hybrid Search** — ChromaDB for source filtering
- **Admin Dashboard** — SQLAdmin for database management

### Frontend (Streamlit)

- **Multiple Chat Modes** — Simple, History, RAG, Filtered RAG
- **Streaming UI** — Real-time token display
- **User Sessions** — Per-user conversation tracking

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /test_endpoint` | Test LLM connectivity |
| `POST /simple/stream` | Basic streaming chat |
| `POST /history/stream` | Chat with conversation memory |
| `POST /rag/stream` | RAG with vector retrieval |
| `POST /filtered_rag/stream` | RAG with hybrid search |

## Backend Components

### Streaming Response

```python
async def generate_stream(response: Generator, callback=None):
    full_response = ""
    for chunk in response:
        data = f"chunk: {chunk.choices[0].delta.content}"
        full_response += chunk.choices[0].delta.content
        yield f"data: {data}\n\n"
    
    if callback:
        callback(full_response)  # Save to DB
    yield "data: [DONE]\n\n"
```

### Chat with History

```python
@app.post("/history/stream")
async def history_stream(request: UserRequest, db: Session = Depends(get_db)):
    # Save user message
    add_message(db, request.question, MessageType.USER, request.username)
    
    # Get conversation history
    chat_history = get_user_chat_history(db, request.username)
    
    # Stream response with callback to save assistant reply
    response = streaming_with_history(chat_history)
    return StreamingResponse(
        generate_stream(response, callback=lambda r: add_message(db, r, MessageType.ASSISTANT, request.username)),
        media_type="text/event-stream"
    )
```

### RAG Pipeline

```python
def streaming_with_rag(chat_history, hybrid_search=False):
    # Condense chat history to standalone query
    user_query = client.chat_completion(
        messages=[{'role': 'user', 'content': HISTORY_SUMMARIZER.format(chat_history=...)}]
    ).choices[0].message.content
    
    # Retrieve relevant documents
    documents = indexer.search(user_query, hybrid_search=hybrid_search)
    
    # Generate response with context
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT.format(context=documents)},
        {'role': 'user', 'content': user_query}
    ]
    return client.chat_completion(messages=messages, stream=True)
```

### Vector Indexing

```python
class DataIndexer:
    def __init__(self, index_name='langchain-repo'):
        self.embedding_client = openai_client
        self.pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        self.index = self.pinecone_client.Index(index_name)
        self.source_index = self.get_source_index()  # ChromaDB for filtering
    
    def search(self, text_query, top_k=5, hybrid_search=False):
        # Embed query
        vector = self.embedding_client.embeddings.create(
            input=text_query, model="text-embedding-3-small"
        ).data[0].embedding
        
        # Optional: Filter by relevant sources
        if hybrid_search:
            sources = self.source_index.query(query_embeddings=[vector], n_results=50)
            filter = {"source": {"$in": sources}}
        
        # Query Pinecone
        results = self.index.query(vector=vector, top_k=top_k, filter=filter)
        return [CodeDoc.model_validate(r["metadata"]) for r in results["matches"]]
```

## Database Schema

### User Table
```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    messages = relationship("Message", back_populates="user")
```

### Message Table
```python
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    type = Column(Enum(MessageType))  # USER or ASSISTANT
    timestamp = Column(DateTime)
    user_id = Column(Integer, ForeignKey("users.id"))
```

## Project Structure

```
├── Backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── test_client.py
│   └── app/
│       ├── main.py           # FastAPI endpoints
│       ├── schemas.py        # Pydantic models
│       ├── models.py         # SQLAlchemy models
│       ├── database.py       # DB connection
│       ├── crud.py           # Database operations
│       ├── clients.py        # LLM client
│       ├── data_indexing.py  # Vector search
│       ├── admin.py          # SQLAdmin views
│       └── methods/
│           ├── simple.py     # Basic chat
│           ├── history.py    # Chat with memory
│           └── rag.py        # RAG implementation
└── Frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── src/
        ├── app.py            # Streamlit main
        ├── api.py            # Backend client
        ├── chat.py           # Chat components
        └── pages/
            ├── simple_page.py
            ├── history_page.py
            ├── rag_page.py
            └── filtered_rag_page.py
```

## Deployment

### Local Development

```bash
# Backend
cd Backend
uvicorn app.main:app --reload --port 8000

# Frontend
cd Frontend
streamlit run src/app.py
```

### Docker (HuggingFace Spaces)

Backend Dockerfile:
```dockerfile
FROM python:3.12
RUN useradd -m -u 1000 user
USER user
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Environment Variables

```bash
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key  # For embeddings
```

## Data Ingestion

Index documents for RAG:

```python
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load repository
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./code_data/",
    branch="master"
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=10000, 
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# Index
indexer = DataIndexer()
indexer.index_data([CodeDoc(content=d.page_content, source=d.metadata['source']) for d in chunks])
```

## Key Technologies

- **FastAPI** — Async web framework
- **SQLAlchemy** — ORM for database operations
- **Pinecone** — Vector database for similarity search
- **ChromaDB** — Local vector store for source filtering
- **OpenAI Embeddings** — text-embedding-3-small
- **Streamlit** — Frontend UI framework
- **SQLAdmin** — Admin dashboard

## References

- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [FastAPI Streaming](https://fastapi.tiangolo.com/advanced/custom-response/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)


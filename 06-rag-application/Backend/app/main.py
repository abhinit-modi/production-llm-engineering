import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Generator, Callable
from contextlib import asynccontextmanager
from sqladmin import Admin
from typing import Optional

from schemas import UserRequest
from clients import client
from database import SessionLocal, engine, get_db, create_tables
from methods.simple import simple_streaming
from methods.history import streaming_with_history
from methods.rag import streaming_with_rag
from crud import get_user_chat_history, add_message, MessageType
from admin import UserAdmin, MessageAdmin

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    admin = Admin(
        app, 
        engine, 
        title="Admin",
        base_url="/admin",
        templates_dir="templates"
    )
    admin.add_view(UserAdmin)
    admin.add_view(MessageAdmin)
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="RAG App",
    lifespan=lifespan
)


async def generate_stream(response: Generator, callback: Optional[Callable[[str], None]] = None):
    full_response = ""
    try:
        for chunk in response:
            # TODO: Implement the function in to send the streaming chunks of data. 
            data = f"chunk: {chunk.choices[0].delta.content}"
            full_response += chunk.choices[0].delta.content
            yield f"data: {data}\n\n"
        if callback:
            callback(full_response)
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield "data: [DONE]\n\n"


@app.post("/test_endpoint")
async def test_endpoint(request: UserRequest):
    try:
        response = client.chat_completion(
            messages=[{'role': 'user', 'content': request.question}],
        )
        return {"status": "success", "response": str(response.choices[0].message.content)}
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simple/stream")
async def simple_stream(request: UserRequest):
    try:
        response = simple_streaming(request.question)
        return StreamingResponse(
            generate_stream(response),
            media_type="text/event-stream",
        )
    except Exception as e:
        print(f"Error in simple_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/stream")
async def history_stream(request: UserRequest, db: Session = Depends(get_db)):  
    # TODO:  Let's implement the "/history/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The new user question is saved in the database by using add_message
    # - The user request is used to pull the chat history of the user
    # - We use the chat history with the streaming_with_history function.
    # - Add a callback to capture the response from the assistant
    try:
        new_message = add_message(db, request.question, MessageType.USER, request.username)
        chat_history = get_user_chat_history(db, request.username)
        response = streaming_with_history(chat_history)
        return StreamingResponse(
            generate_stream(response, callback=lambda full_response: add_message(db, full_response, MessageType.ASSISTANT, request.username)),
            media_type="text/event-stream",
        )
    except Exception as e:
        print(f"Error in history_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/stream")
async def rag_stream(request: UserRequest, db: Session = Depends(get_db)):  
    # TODO: Let's implement the "/rag/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The new user question is saved in the database by using add_message
    # - The user request is used to pull the chat history of the user
    # - We use the chat history with the streaming_with_rag function.
    # - Add a callback to capture the response from the assistant
    try:
        add_message(db, request.question, MessageType.USER, request.username)
        chat_history = get_user_chat_history(db, request.username)
        response = streaming_with_rag(chat_history)
        return StreamingResponse(
            generate_stream(response, callback=lambda full_response: add_message(db, full_response, MessageType.ASSISTANT, request.username)),
            media_type="text/event-stream",
        )
    except Exception as e:
        print(f"Error in streaming rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/filtered_rag/stream")
async def filtered_rag_stream(request: UserRequest, db: Session = Depends(get_db)):  
    # TODO: Let's implement the "/filtered_rag/stream" endpoint. The endpoint should follow those steps:
    # - The endpoint receives the request
    # - The new user question is saved in the database by using add_message
    # - The user request is used to pull the chat history of the user
    # - We use the chat history with the streaming_with_rag function with hybrid_search = True.
    # - Add a callback to capture the response from the assistant
    try:
        add_message(db, request.question, MessageType.USER, request.username)
        chat_history = get_user_chat_history(db, request.username)
        response = streaming_with_rag(chat_history, hybrid_search=True)
        return StreamingResponse(
            generate_stream(response, callback=lambda full_response: add_message(db, full_response, MessageType.ASSISTANT, request.username)),
            media_type="text/event-stream",
        )
    except Exception as e:
        print(f"Error in filtered streaming rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", reload=True,  port=8000)
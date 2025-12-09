from sqlalchemy.orm import Session
from models import User, Message, MessageType
from typing import Any

def commit(db: Session, instance: Any):
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance

def get_or_create_user(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        user = commit(db, user)
    return user

def add_message(db: Session, message: str, message_type: MessageType, username: str):
    user = get_or_create_user(db, username)
    db_message = Message(
        message=message,
        type=message_type,
        user_id=user.id,
        user=user
    )
    db_message = commit(db, db_message)
    return db_message

def get_user_chat_history(db: Session, username: str, limit: int = 20) -> list[dict[str, str]]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []
    chat_history = db.query(Message).filter(Message.user_id == user.id)
    chat_history = chat_history.order_by(Message.timestamp.desc()).limit(20)
    return [
        {
            'role': 'user' if message.type == MessageType.USER else 'assistant',
            'content': message.message
        } for message in chat_history]
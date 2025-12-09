from pydantic import BaseModel


# Pydantic model for chat request validation
class UserRequest(BaseModel):
    username: str
    question: str



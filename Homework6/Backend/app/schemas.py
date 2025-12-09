from pydantic import BaseModel


# TODO: let's create a UserRequest data model with a question and username attribute. 
# This will be used to parse the input request.
class UserRequest(BaseModel):
    username: str
    question: str



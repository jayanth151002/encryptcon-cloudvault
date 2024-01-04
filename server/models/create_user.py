from pydantic import BaseModel


class CreateUser(BaseModel):
    username: str
    user_type: str

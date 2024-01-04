from pydantic import BaseModel


class Verify2Fa(BaseModel):
    id: str
    code: str
    password: str

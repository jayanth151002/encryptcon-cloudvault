from pydantic import BaseModel


class Transaction(BaseModel):
    Time: int
    Source: str
    Target: str
    Location: str
    Type: str
    Amount: float
    Labels: int

from pydantic import BaseModel


class LoanApprover(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: int
    Credit_History: int
    Property_Area: str

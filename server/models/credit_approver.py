from pydantic import BaseModel


class CreditApprover(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: int
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_MOBIL: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: int
    Card_Issued: int

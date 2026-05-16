from typing import Optional

from pydantic import BaseModel


class TransactionInput(BaseModel):
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: Optional[int] = None

    model_config = {"extra": "ignore"}


class PredictionResponse(BaseModel):
    trans_num: str
    fraud_probability: float
    is_fraud: bool
    threshold: float

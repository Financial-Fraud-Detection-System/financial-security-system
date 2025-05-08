from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """
    A financial transaction.
    """

    cc_num: str = Field(..., description="Credit card number")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., description="Transaction amount")
    first: str = Field(..., description="First name of the cardholder")
    last: str = Field(..., description="Last name of the cardholder")
    gender: Literal["M", "F"] = Field(..., description="Gender of the cardholder")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    zip: str = Field(..., description="ZIP code")
    lat: float = Field(..., description="Latitude of the cardholder's location")
    long: float = Field(..., description="Longitude of the cardholder's location")
    city_pop: int = Field(..., description="Population of the city")
    job: str = Field(..., description="Job title of the cardholder")
    dob: date = Field(..., description="Date of birth of the cardholder")
    unix_time: int = Field(..., description="Unix timestamp of the transaction")
    merch_lat: float = Field(..., description="Latitude of the merchant's location")
    merch_long: float = Field(..., description="Longitude of the merchant's location")

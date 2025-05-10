from pydantic import BaseModel, Field, IPvAnyAddress, PositiveFloat, PositiveInt, constr


class OnlineTransaction(BaseModel):
    acc_num: str = Field(..., description="Unique account identifier")
    amt: PositiveFloat = Field(..., description="Transaction amount (must be > 0)")
    state: str = Field(
        ...,
        description="US state/territory abbreviation (e.g., 'NY')",
        min_length=2,
        max_length=2,
    )
    zip: str = Field(..., description="5-digit ZIP code", min_length=5, max_length=5)
    city_pop: PositiveInt = Field(..., description="Population of the transaction city")
    trans_num: str = Field(..., description="Unique transaction identifier")
    unix_time: PositiveInt = Field(
        ..., description="Unix timestamp (seconds since 1970-01-01 UTC)"
    )
    ip_address: IPvAnyAddress = Field(..., description="Originating IP address")

    class Config:
        anystr_strip_whitespace = True
        frozen = True  # make instances hashable / immutable
        schema_extra = {
            "example": {
                "acc_num": "acc123",
                "amt": 5000.0,
                "state": "NY",
                "zip": "10001",
                "city_pop": 1000000,
                "trans_num": "t1",
                "unix_time": 1625097600,
                "ip_address": "192.168.1.1",
            }
        }

from pydantic import BaseModel


class DummyModel(BaseModel):
    attribute_1: str
    attribute_2: int

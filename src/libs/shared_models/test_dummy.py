import pytest
from dummy import DummyModel
from pydantic import ValidationError


def test_create_dummy_model_success():
    dummy = DummyModel(**{"attribute_1": "test", "attribute_2": 42})
    assert dummy.attribute_1 == "test"
    assert dummy.attribute_2 == 42


def test_create_dummy_model_validation_error():
    with pytest.raises(ValidationError):
        DummyModel(**{"attribute_1": "test", "attribute_2": "invalid_int"})

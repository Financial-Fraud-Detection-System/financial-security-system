import logging

from libs.shared_models.dummy import DummyModel


def dummy_handler(key: str, value: dict):
    """
    Handle incoming dummy and process them.

    :param key: The key of the incoming dummy.
    :param value: The value of the incoming dummy.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Received dummy: key={key}, value={value}")
    try:
        dummy = DummyModel(**value)
    except ValueError:
        logger.error(f"Error deserializing dummy value: {value}")
        return
    logger.info(
        f"Processing dummy for key: {key}, attributes: {dummy.attribute_1} {dummy.attribute_2}"
    )

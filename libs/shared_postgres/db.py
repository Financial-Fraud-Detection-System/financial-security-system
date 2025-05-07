from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_session_maker(database_url: str):
    """
    Returns a session local object for database interactions.
    """
    engine = create_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)

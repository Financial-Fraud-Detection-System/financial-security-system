from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def initialize_database(Base, database_url: str):
    """
    Initializes the database by creating all tables.
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)


def get_session_maker(database_url: str):
    """
    Returns a session local object for database interactions.
    """
    engine = create_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)

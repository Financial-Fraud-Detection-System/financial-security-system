import os

SQLALCHEMY_DATABASE_URI = os.getenv(
    "SUPERSET_DATABASE_URI", "sqlite:////app/superset_home/superset.db"
)

FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}

ENABLE_PROXY_FIX = True

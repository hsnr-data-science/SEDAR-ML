from dotenv import load_dotenv
import os

load_dotenv()

def _parse_bool(value):
    return str(value).lower() in ['true', '1', 't', 'y', 'yes']

class Config(object):
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = os.getenv("PORT", 8050)
    DEBUG = _parse_bool(os.getenv("DEBUG", False))
    VERBOSE = _parse_bool(os.getenv("VERBOSE", True))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
    WORKING_DIR = os.path.expanduser(os.getenv("WORKING_DIR", "~/.assistml/dashboard"))
    # SAVE_UPLOADS = _parse_bool(os.getenv("SAVE_UPLOADS", False))

    MONGO_HOST = os.getenv("MONGO_HOST")
    MONGO_PORT = int(os.getenv("MONGO_PORT"))
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    MONGO_DB = os.getenv("MONGO_DB", "assistml")
    MONGO_TLS = _parse_bool(os.getenv("MONGO_TLS", False))

    assert BACKEND_BASE_URL is not None, "BACKEND_BASE_URL must be set"

    assert MONGO_HOST is not None, "MONGO_HOST must be set"
    assert MONGO_PORT is not None, "MONGO_PORT must be set"
    assert MONGO_USER is not None, "MONGO_USER must be set"
    assert MONGO_PASS is not None, "MONGO_PASS must be set"


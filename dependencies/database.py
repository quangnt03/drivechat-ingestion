# Dependency to get database session
from services.db import DatabaseService
import os

def get_db():
    db = DatabaseService(os.getenv("DATABASE_URL"))
    return db.session
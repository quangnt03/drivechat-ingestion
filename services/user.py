from sqlalchemy.orm import Session
from models.user import User
from typing import Optional
import logging
class UserService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)
        
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    def create_user(self, email: str) -> User:
        user = User(email=email, display_name=email.split('@')[0])
        self.db.add(user)
        self.db.flush()
        self.db.expunge(user)
        return user
    

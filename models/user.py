from .common import Base, BaseConfig
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class User(Base):
    """SQLAlchemy model for users table."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    display_name = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    messages = relationship("Message", back_populates="user")
    owned_items = relationship("Item", back_populates="owner")

    def __repr__(self):
        return f"<User(id={self.id}, display_name='{self.display_name}')>"

# Pydantic models for request/response validation
class UserBase(BaseModel):
    display_name: str
    email: str
    active: Optional[bool] = True

    class Config(BaseConfig):
        pass

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    display_name: Optional[str] = None
    email: Optional[str] = None
    active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config(BaseConfig):
        pass


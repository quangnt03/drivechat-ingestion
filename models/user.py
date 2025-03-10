from .base import Base
import uuid
from sqlalchemy import Column, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class User(Base):
    """SQLAlchemy model for users table."""
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)

    # Relationships
    owned_items = relationship("Item", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"

# Pydantic models for request/response validation
class UserBase(BaseModel):
    display_name: str
    email: str
    active: Optional[bool] = True

    class Config:
        from_attributes = True

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    display_name: Optional[str] = None
    email: Optional[str] = None
    active: Optional[bool] = None

class UserResponse(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


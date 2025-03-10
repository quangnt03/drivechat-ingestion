import uuid
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from .base import Base

class Item(Base):
    """
    Model representing a document item in the system.
    """
    __tablename__ = "items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    active = Column(Boolean, default=True)

    # Define relationships
    owner = relationship("User", back_populates="owned_items")
    conversation = relationship("Conversation", back_populates="items")
    embeddings = relationship("Embedding", back_populates="item", cascade="all, delete-orphan", lazy="dynamic")

    def __repr__(self):
        return f"<Item(id={self.id}, file_name='{self.file_name}')>" 
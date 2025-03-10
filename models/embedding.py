from sqlalchemy import Column, Integer, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID
from .base import Base
import uuid

class Embedding(Base):
    """
    Model representing text embeddings with vector support.
    """
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    page = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536)) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Define relationship with Item model
    item = relationship("Item", back_populates="embeddings")
    messages = relationship("Message", back_populates="source_embedding")

    # Composite unique constraint and table configuration
    __table_args__ = (
        UniqueConstraint('item_id', 'page', name='uix_item_page'),
    )

    def __repr__(self):
        return f"<Embedding(id={self.id}, item_id={self.item_id}, page={self.page})>"

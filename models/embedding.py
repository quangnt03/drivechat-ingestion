from sqlalchemy import Column, Integer, DateTime, ForeignKey, Text, UniqueConstraint, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from .base import Base

class Embedding(Base):
    """
    Model representing text embeddings with vector support.
    """
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String, ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    page = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536)) 
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())

    # Define relationship with Item model
    item = relationship("Item", back_populates="embeddings")

    # Composite unique constraint and table configuration
    __table_args__ = (
        UniqueConstraint('item_id', 'page', name='uix_item_page'),
    )

    def __repr__(self):
        return f"<Embedding(id={self.id}, item_id={self.item_id}, page={self.page})>"

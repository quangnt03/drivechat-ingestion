from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class ConversationCreate(BaseModel):
    title: str
    context: str

class ConversationResponse(BaseModel):
    id: UUID
    title: str
    user_id: UUID
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True 
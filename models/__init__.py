from .base import Base
from .user import User
from .conversation import Conversation
from .item import Item
from .embedding import Embedding
from .message import Message

# Define the order of table creation
__all__ = [
    'Base',
    'User',           # No foreign key dependencies
    'Conversation',   # Depends on User
    'Item',          # Depends on User and Conversation
    'Embedding',     # Depends on Item
    'Message'        # Depends on User, Conversation, and Embedding
]

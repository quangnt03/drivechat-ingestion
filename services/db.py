from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Optional
import logging
from models import Base, User, Conversation, Item, Embedding, Message
from llama_index.core.schema import Node
from datetime import datetime
from uuid import UUID

class DatabaseService:
    def __init__(self, db_url: str):
        """
        Initialize database connection and create tables.
        
        Args:
            db_url (str): Database connection URL
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine = create_engine(db_url)
        
        # Create all tables before creating the session
        try:
            # Enable pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            
            if not self.verify_tables():
                # Drop all tables in reverse dependency order
                Base.metadata.drop_all(bind=self.engine)
                
                # Create tables in dependency order
                tables = [
                    User.__table__,
                    Conversation.__table__,
                    Item.__table__,
                    Embedding.__table__,
                    Message.__table__
                ]
                
                # Create tables one by one
                for table in tables:
                    try:
                        table.create(self.engine)
                        self.logger.info(f"Created table: {table.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to create table {table.name}: {str(e)}")
                        raise
            
            # Verify tables were created
            if self.verify_tables():
                self.logger.info("Database tables created successfully")
            else:
                raise Exception("Failed to verify table creation")
            
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {str(e)}")
            raise
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.session = self.SessionLocal()

    def insert_document_with_embeddings(
        self, 
        nodes: List[Node], 
        metadata: Dict,
        owner_id: UUID,
        conversation_id: UUID
    ) -> Optional[Item]:
        try:
            session = self.SessionLocal()
            try:
                # Create Item
                item = Item(
                    file_name=metadata['file_name'],
                    mime_type=metadata['mime_type'],
                    uri=metadata['uri'],
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                    last_updated=datetime.now()
                )
                session.add(item)
                session.flush()

                # Create Embeddings
                for node in nodes:
                    page = int(node.extra_info.get('page_label', -1))
                    embedding = Embedding(
                        item_id=item.id,
                        page=page,
                        chunk_text=node.get_content(),
                        embedding=node.embedding,
                    )
                    session.add(embedding)
                    session.flush()
                session.commit()
                self.logger.info(f"Successfully inserted document {item.file_name} with {len(nodes)} chunks")
                return item
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to insert document: {str(e)}")
                raise
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            return None
        
    def get_or_create_user(self, email: str, display_name: Optional[str] = None) -> User:
        """
        Get existing user or create new one.
        
        Args:
            email (str): User's email address
            display_name (Optional[str]): User's display name
            
        Returns:
            User: Existing or new user
        """
        try:
            session = self.SessionLocal()
            
            try:
                # Try to find existing user
                user = session.query(User).filter(User.email == email).first()
                
                if not user:
                    # Create new user if not found
                    user = User(
                        email=email,
                        display_name=display_name or email.split('@')[0],
                        active=True
                    )
                    session.add(user)
                    session.commit()
                    self.logger.info(f"Created new user: {email}")
                
                return user
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to get/create user: {str(e)}")
                raise
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Database error in get_or_create_user: {str(e)}")
            raise
        
    def verify_tables(self) -> bool:
        """
        Verify that all required tables exist in the database.
        
        Returns:
            bool: True if all tables exist, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Check if tables exist in correct order
                tables = ['users', 'conversations', 'items', 'embeddings', 'messages']
                for table in tables:
                    result = conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = '{table}'
                        );
                    """))
                    exists = result.scalar()
                    if not exists:
                        self.logger.error(f"Table '{table}' does not exist")
                        return False
                
                self.logger.info("All required tables exist")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to verify tables: {str(e)}")
            return False


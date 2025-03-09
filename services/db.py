from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Optional
from models.item import Item
from models.embedding import Embedding
from models.base import Base
from llama_index.core.schema import Node
from datetime import datetime
from sqlalchemy.sql import text
import logging

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
            # Drop existing tables and recreate thems
            
            # Enable pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            
            # Create all tables defined in the models
            Base.metadata.create_all(bind=self.engine)
            
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

    def insert_document_with_embeddings(self, nodes: List[Node], metadata: Dict) -> Optional[Item]:
        try:
            # Create new session for this operation
            session = self.SessionLocal()
            
            try:
                item = Item(
                    id=metadata['id'],
                    file_name=metadata['file_name'],
                    mime_type=metadata['mime_type'],
                    uri=metadata['uri'],
                    owner=metadata['owner'],
                    conversation_id=metadata['conversation_id'],
                    last_updated=datetime.now(),
                    active=True
                )
                session.add(item)
                session.flush()

                for node in nodes:
                    page = int(node.extra_info.get('page_label', -1))
                    embedding = Embedding(
                        item_id=item.id,
                        page=page,
                        chunk_text=node.get_content(),
                        embedding=node.embedding,
                        last_updated=datetime.now()
                    )
                    session.add(embedding)
                
                session.commit()
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

    def verify_tables(self) -> bool:
        """
        Verify that all required tables exist in the database.
        
        Returns:
            bool: True if all tables exist, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Check if tables exist
                tables = ['items', 'embeddings']
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


import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models.base import Base
from models.item import Item
from models.embedding import Embedding
from typing import List, Dict, Optional
from llama_index.core.schema import Node
from datetime import datetime

class DatabaseManager:
    def __init__(self, connection_string: str):
        """
        Initialize database connection with SQLAlchemy.
        
        Args:
            connection_string (str): PostgreSQL connection string
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(connection_string)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            self._initialize_database()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _initialize_database(self):
        """Create necessary tables, extensions, and indexes if they don't exist."""
        try:
            with self.engine.connect() as conn:
                # Enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # Create tables if they don't exist
                Base.metadata.create_all(bind=self.engine)
                
                # Create indexes for better performance
                conn.execute(text("""
                    -- Create GiST index for vector similarity search if not exists
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c 
                            JOIN pg_namespace n ON n.oid = c.relnamespace 
                            WHERE c.relname = 'embeddings_embedding_idx'
                        ) THEN
                            CREATE INDEX embeddings_embedding_idx 
                            ON embeddings 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        END IF;
                    END $$;
                    
                    -- Create index on items.active if not exists
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = 'items_active_idx'
                        ) THEN
                            CREATE INDEX items_active_idx ON items(active);
                        END IF;
                    END $$;
                    
                    -- Create index on items.conversation_id if not exists
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = 'items_conversation_id_idx'
                        ) THEN
                            CREATE INDEX items_conversation_id_idx ON items(conversation_id);
                        END IF;
                    END $$;
                    
                    -- Create index on embeddings.item_id if not exists
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = 'embeddings_item_id_idx'
                        ) THEN
                            CREATE INDEX embeddings_item_id_idx ON embeddings(item_id);
                        END IF;
                    END $$;
                """))
                
                # Commit all changes
                conn.commit()
                
            self.logger.info("Database initialized successfully with all required tables and indexes")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def insert_document(self, nodes: List[Node], metadata: Dict) -> Optional[Item]:
        """
        Insert a document and its embeddings into the database.
        
        Args:
            nodes (List[Node]): List of LlamaIndex nodes containing text and embeddings
            metadata (Dict): Document metadata containing:
                - id: Document ID
                - file_name: Name of the file
                - mime_type: MIME type
                - uri: Document URI
                - owner: Document owner
                - conversation_id: Conversation ID
                
        Returns:
            Optional[Item]: Created Item object or None if failed
        """
        try:
            session = self.SessionLocal()
            
            try:
                # Create Item
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
                session.flush()  # Get the item ID
                
                # Create Embeddings for each node
                for node in nodes:
                    # Extract page number from node metadata
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
                self.logger.info(f"Successfully inserted document {item.file_name} with {len(nodes)} chunks")
                return item
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to insert document: {str(e)}")
                return None
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            return None

    def get_document(self, doc_id: str) -> Optional[Item]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Optional[Item]: Item object or None if not found
        """
        try:
            session = self.SessionLocal()
            return session.query(Item).filter(Item.id == doc_id).first()
        finally:
            session.close()

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 5,
        active_only: bool = True
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            limit (int): Maximum number of results
            active_only (bool): Whether to search only active documents
            
        Returns:
            List[Dict]: List of similar chunks with their document metadata
        """
        try:
            session = self.SessionLocal()
            
            # Build the query with active document filter if needed
            active_filter = "AND i.active = true" if active_only else ""
            
            results = session.execute(text("""
                SELECT 
                    i.id as doc_id,
                    i.file_name,
                    i.uri,
                    i.owner,
                    e.chunk_text,
                    e.page,
                    1 - (e.embedding <=> :query_embedding) as similarity
                FROM embeddings e
                JOIN items i ON i.id = e.item_id
                WHERE e.embedding IS NOT NULL
                """ + active_filter + """
                ORDER BY e.embedding <=> :query_embedding
                LIMIT :limit
            """), {
                'query_embedding': query_embedding,
                'limit': limit
            })
            
            return [dict(row) for row in results]
            
        finally:
            session.close()
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose()

# Initialize the service

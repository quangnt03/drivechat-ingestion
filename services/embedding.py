from typing import Dict, List, Optional, Tuple
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import TokenTextSplitter
import logging

class EmbeddingService:
    def __init__(self, openai_api_key: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the embedding service.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings
            chunk_size (int): Size of text chunks in tokens
            chunk_overlap (int): Number of overlapping tokens between chunks
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def _load_document(self, file_path: str) -> Optional[Document]:
        """
        Load a single document using LlamaIndex.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            Optional[Document]: Loaded document or None if failed
        """
        try:
            reader = SimpleDirectoryReader(
                input_files=[file_path]
            )
            docs = reader.load_data()
            return docs[0] if docs else None
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {str(e)}")
            return None
            
    def _load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all documents from a directory using LlamaIndex.
        
        Args:
            directory_path (str): Path to the directory
            
        Returns:
            List[Document]: List of loaded documents
        """
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=True
            )
            return reader.load_data()
        except Exception as e:
            self.logger.error(f"Error loading directory {directory_path}: {str(e)}")
            return []
            
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the text splitter.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            return self.text_splitter.split_text(text)
        except Exception as e:
            self.logger.error(f"Error splitting text into chunks: {str(e)}")
            return [text]  # Return full text as single chunk if splitting fails
            
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a text string.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        try:
            embedding = await self.embed_model.aget_text_embedding(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return None
            
    async def embed_chunks(self, chunks: List[str]) -> List[Tuple[str, List[float]]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            List[Tuple[str, List[float]]]: List of (chunk_text, embedding) tuples
        """
        results = []
        for chunk in chunks:
            try:
                embedding = await self.embed_model.aget_text_embedding(chunk)
                if embedding:
                    results.append((chunk, embedding))
            except Exception as e:
                self.logger.error(f"Error embedding chunk: {str(e)}")
                continue
        return results
            
    async def embed_file(self, file_path: str) -> Optional[List[Tuple[str, List[float]]]]:
        """
        Generate embeddings for chunks of a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Optional[List[Tuple[str, List[float]]]]: List of (chunk_text, embedding) tuples
        """
        doc = self._load_document(file_path)
        if not doc:
            return None
            
        try:
            # Split document into chunks
            chunks = self._chunk_text(doc.text)
            self.logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            return await self.embed_chunks(chunks)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
            
    async def embed_folder(self, folder_path: str) -> Dict[str, List[Tuple[str, List[float]]]]:
        """
        Generate embeddings for chunks of all files in a folder.
        
        Args:
            folder_path (str): Path to the folder
            
        Returns:
            Dict[str, List[Tuple[str, List[float]]]]: Dictionary mapping file paths 
                to lists of (chunk_text, embedding) tuples
        """
        docs = self._load_directory(folder_path)
        results = {}
        
        for doc in docs:
            try:
                file_path = doc.metadata.get('file_path', '')
                if file_path:
                    # Split document into chunks
                    chunks = self._chunk_text(doc.text)
                    self.logger.info(f"Split {file_path} into {len(chunks)} chunks")
                    
                    # Generate embeddings for chunks
                    chunk_embeddings = await self.embed_chunks(chunks)
                    if chunk_embeddings:
                        results[file_path] = chunk_embeddings
                        
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
                
        return results

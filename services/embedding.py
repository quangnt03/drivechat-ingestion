from typing import Dict, List
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import SentenceSplitter
import logging

class EmbeddingService:
    def __init__(self, openai_api_key: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the embedding service.
        
        Args:
            openai_api_key (str): OpenAI API key for embeddings
            chunk_size (int): Size of text chunks in tokens
            chunk_overlap (int): Number of overlapping tokens between chunks
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def process_document(self, file_path: str) -> List[Dict]:
        """
        Process a document: load, split, and embed.
        
        Args:
            file_path (str): Path to the document
        
        Returns:
            List[Dict]: List of chunks with their embeddings
        """
        try:
            # 1. Load document
            loader = SimpleDirectoryReader(file_path)
            documents = loader.load_data()
            nodes = self.text_splitter.get_nodes_from_documents(documents)
            
            # 2. Generate embeddings for each chunk
            for node in nodes:
                node.embedding = await self.embed_model.aget_text_embedding(
                    node.get_content()
                )
                
            return nodes
            
        except Exception as e:
            raise Exception(f"Failed to process document: {str(e)}")

    def metadata_handler(self, metadata: Dict, owner: str, conversation_id: str) -> Dict:
        """
        Extract and format metadata from Google Drive file.
        
        Args:
            metadata (Dict): Raw metadata from Google Drive
            
        Returns:
            Dict: Formatted metadata
        """
        return {
            "file_name": metadata.get('name'),
            "id": metadata.get('id'),
            "uri": metadata.get('webViewLink'),
            "mime_type": metadata.get('mimeType'),
            "owner": owner,
            "conversation_id": conversation_id
        }
        
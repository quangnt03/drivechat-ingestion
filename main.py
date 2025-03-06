import os
from tempfile import TemporaryDirectory
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
import tempfile
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from dependencies.security import validate_token
load_dotenv(find_dotenv())
from utils.gdrive import GoogleDriveClient  # noqa: E402
from services.db import DatabaseService  # noqa: E402


app = FastAPI()

# Configuration
CREDENTIALS_PATH = os.path.join(os.getcwd(), ".credentials", "service-account-key.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONNECTION = os.getenv("DATABASE_URL")

class DriveUrl(BaseModel):
    driver_id: str
    conversation_id: str


gclient = GoogleDriveClient(CREDENTIALS_PATH)
db_service = DatabaseService(DB_CONNECTION)
embedding_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


@app.post("/api/v1/upload")
async def upload_file(file: DriveUrl, user = Depends(validate_token)):
    owner = user['UserAttributes'][0]['Value']
    verified = user['UserAttributes'][1]['Value']
    if not verified:
        raise HTTPException(status_code=403, detail="User is not verified")
    try:
        file_id, file_type = gclient.get_gdrive_id(file.driver_id)
        if file_id is None:
            raise HTTPException(status_code=400, detail="Invalid drive url")

        # Create temporary directory
        with TemporaryDirectory(
            prefix="gdrive_",
            dir=tempfile.gettempdir()
        ) as temp_dir:
            # Download file(s)
            if file_type != 'file':
                raise HTTPException(status_code=400, detail="Folder upload is not supported yet")
            
            result = gclient.download_file(file_id, temp_dir)

            if not result['success']:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to download from Google Drive: {result.get('error', 'Unknown error')}"
                )
                
            try:
                # Process files
                metadata = result.get('files', None)
                if not metadata:
                    raise HTTPException(status_code=500, detail="No file metadata received")

                # Process document and get chunks with embeddings
                nodes = await process_document(temp_dir, embedding_model)
                formatted_metadata = metadata_handler(metadata)
                formatted_metadata['conversation_id'] = file.conversation_id
                formatted_metadata['owner'] = owner

                # Insert document and chunks into database
                item = db_service.insert_document_with_embeddings(nodes, formatted_metadata)
                if not item:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to store document in database"
                    )

                # Format response
                return {
                    "message": "Files processed and stored successfully",
                    "processed_files": 1,
                    "conversation_id": file.conversation_id,
                    "owner": owner,
                    **formatted_metadata,
                    "nodes": len(nodes),
                }                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process documents: {str(e)}"
                )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload process failed: {str(e)}"
        )

async def process_document(file_path: str, embedding_model: OpenAIEmbedding) -> List[Dict]:
    """
    Process a document: load, split, and embed.
    
    Args:
        file_path (str): Path to the document
        embedding_model: OpenAI embedding model instance
    
    Returns:
        List[Dict]: List of chunks with their embeddings
    """
    try:
        # 1. Load document
        loader = SimpleDirectoryReader(file_path)
        documents = loader.load_data()
        text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
        nodes = text_splitter.get_nodes_from_documents(documents)
        
        # 2. Generate embeddings for each chunk
        for node in nodes:
            node.embedding = await embedding_model.aget_text_embedding(node.get_content())
            
        return nodes
        
    except Exception as e:
        raise Exception(f"Failed to process document: {str(e)}")

def metadata_handler(metadata: Dict) -> Dict:
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
        "mime_type": metadata.get('mimeType')
    }
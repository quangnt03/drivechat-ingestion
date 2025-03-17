import os
from tempfile import TemporaryDirectory
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from dotenv import load_dotenv, find_dotenv
from dependencies.security import validate_token
load_dotenv(find_dotenv())
from utils.gdrive import GoogleDriveClient  # noqa: E402
from services.db import DatabaseService  # noqa: E402
from services.embedding import EmbeddingService # noqa: E402
from services.user import UserService # noqa: E402
from uuid import UUID
from services.conversation import ConversationService


app = FastAPI()

# Configuration
CREDENTIALS_PATH = os.path.join(os.getcwd(), ".credentials", "service-account-key.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONNECTION = os.getenv("DATABASE_URL")

class DriveUrl(BaseModel):
    driver_id: str
    conversation_id: UUID


gclient = GoogleDriveClient(CREDENTIALS_PATH)
db_service = DatabaseService(DB_CONNECTION)
embedding_service = EmbeddingService(OPENAI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/upload")
async def upload_file(file: DriveUrl, user = Depends(validate_token)):
    owner = user['UserAttributes'][0]['Value']
    user_service = UserService(db_service.session)
    user = user_service.get_user_by_email(owner)
    if not user:
        user = user_service.create_user(owner)
    
    conversation_service = ConversationService(db_service.session)
    conversation = conversation_service.get_conversation(file.conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=400, detail="Conversation not found")
    
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
                nodes = await embedding_service.process_document(temp_dir)
                formatted_metadata = embedding_service.metadata_handler(
                    metadata, owner, file.conversation_id
                )

                # Insert document and chunks into database
                item = db_service.insert_document_with_embeddings(
                    nodes=nodes, 
                    metadata=formatted_metadata,
                    owner_id=user.id,
                    conversation_id=file.conversation_id
                )
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


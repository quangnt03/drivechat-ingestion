from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from typing import List, Dict, Optional, Union, Tuple
import os.path
import io
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GoogleDriveClient:
    """A client for interacting with Google Drive API."""
    
    def __init__(self, credentials_path: str, scopes: List[str] = None):
        """
        Initialize the Google Drive client.
        
        Args:
            credentials_path (str): Path to the service account credentials file
            scopes (List[str], optional): List of OAuth scopes. Defaults to readonly scope.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.credentials_path = credentials_path
        self.scopes = scopes or ['https://www.googleapis.com/auth/drive.readonly']
        self.service = self._initialize_service()

    def _initialize_service(self):
        """Initialize and return the Google Drive service."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=self.scopes)
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Drive service: {str(e)}")
            return None

    def _get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Get detailed metadata for a file.
        
        Args:
            file_id (str): The ID of the file
            
        Returns:
            Optional[Dict]: File metadata or None if not found
        """
        try:
            return self.service.files().get(
                fileId=file_id,
                fields="*"  # Request all available fields
            ).execute()
        except Exception as e:
            self.logger.error(f"Error getting metadata for file {file_id}: {str(e)}")
            return None
        
    def get_gdrive_id(self, shareable_link: str):
        """Extract the file or folder ID from a Google Drive shareable link."""
        if "id=" in shareable_link:
            return ((shareable_link.split("id=")[-1].split("&")[0], 'file'))    
        elif "/d/" in shareable_link:
            return ((shareable_link.split("/d/")[-1].split("/")[0], 'file'))
        elif "/folders/" in shareable_link:
            return ((shareable_link.split("/folders/")[-1].split("/")[0], 'folder'))
        return None


    def list_folder_contents(self, folder_id: str) -> List[Dict]:
        """
        List all files and folders in a Google Drive folder.
        
        Args:
            folder_id (str): The ID of the folder to list contents from
            
        Returns:
            List[Dict]: List of file and folder metadata
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return []

        query = f"'{folder_id}' in parents and trashed = false"
        results = []
        page_token = None
        
        try:
            while True:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(*)',  # Request all available fields
                    pageToken=page_token
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
                    
            return results
        except Exception as e:
            self.logger.error(f"Error listing files in folder {folder_id}: {str(e)}")
            return []

    def download_file(self, file_id: str, save_path: str) -> Dict:
        """
        Download a single file from Google Drive and return its metadata.
        
        Args:
            file_id (str): The ID of the file to download
            save_path (str): Path where the file should be saved
            
        Returns:
            Tuple[bool, Optional[Dict]]: (success status, file metadata)
                The metadata dictionary includes all available file information
        """
        if not self.service:
            message = "Google Drive service not initialized"
            self.logger.error(message)
            return {
                'success': False, 
                'files': None, 
                'error': message
            }

        try:
            # Get complete file metadata
            file_metadata = self._get_file_metadata(file_id)
            if not file_metadata:
                return {
                    'success': False, 
                    'files': None, 
                    'error': "File not found"
                }

            request = self.service.files().get_media(fileId=file_id)
            
            # Create a BytesIO object for the download
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            # Download the file
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    self.logger.info(f"Download Progress: {int(status.progress() * 100)}%")
            
            # Save the file
            fh.seek(0)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_name = os.path.join(save_path, file_metadata['name'])
            
            with open(save_name, 'wb') as f:
                f.write(fh.read())
            
            # Add local file information to metadata
            file_metadata.update({
                'local_path': save_name,
                'download_time': datetime.now().isoformat(),
                'local_size': os.path.getsize(save_name)
            })
                
            self.logger.info(f"File '{file_metadata['name']}' downloaded successfully to {save_name}")
            return {
                'success': True, 
                'error': None,
                'files': file_metadata, 
                'message': None
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading file {file_id}: {str(e)}")
            return {
                'success': False, 
                'files': None, 
                'error': str(e)
            }

    def _process_folder(self, 
                      current_folder_id: str, 
                      current_path: str, 
                      depth: int,
                      max_depth: int,
                      file_types: Optional[List[str]],
                      skip_existing: bool,
                      stats: Dict,
                      all_files_metadata: List[Dict]) -> bool:
        """
        Process a folder and its contents for downloading.
        
        Args:
            current_folder_id (str): The ID of the current folder
            current_path (str): Current local path for saving files
            depth (int): Current folder depth
            max_depth (int): Maximum folder depth to traverse
            file_types (List[str], optional): List of allowed file extensions
            skip_existing (bool): Whether to skip existing files
            stats (Dict): Statistics dictionary to update
            all_files_metadata (List[Dict]): List to store all file metadata
            
        Returns:
            bool: True if all operations were successful, False otherwise
        """
        if max_depth != -1 and depth > max_depth:
            return True

        self.logger.info(f"Processing folder at depth {depth}, path: {current_path}")
        items = self.list_folder_contents(current_folder_id)
        
        if not items:
            self.logger.warning(f"No items found in folder {current_folder_id}")
            return True

        success = True
        
        for item in items:
            try:
                item_path = os.path.join(current_path, item['name'])
                stats["files_processed"] += 1
                
                self.logger.debug(f"Processing item: {item['name']} ({item['mimeType']})")
                
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    self.logger.info(f"Found subfolder: {item['name']}")
                    os.makedirs(item_path, exist_ok=True)
                    if not self._process_folder(
                        item['id'], 
                        item_path, 
                        depth + 1,
                        max_depth,
                        file_types,
                        skip_existing,
                        stats,
                        all_files_metadata
                    ):
                        success = False
                else:
                    # Check file type filter
                    if file_types:
                        file_ext = os.path.splitext(item['name'])[1].lower()
                        if file_ext not in file_types:
                            self.logger.info(f"Skipping file {item['name']} - type {file_ext} not in allowed types")
                            stats["files_skipped"] += 1
                            item['download_status'] = 'skipped_file_type'
                            all_files_metadata.append(item)
                            continue
                    
                    # Check if file exists
                    if skip_existing and os.path.exists(os.path.join(current_path, item['name'])):
                        self.logger.info(f"Skipping existing file: {item['name']}")
                        stats["files_skipped"] += 1
                        item['download_status'] = 'skipped_existing'
                        item['local_path'] = os.path.join(current_path, item['name'])
                        all_files_metadata.append(item)
                        continue
                    
                    # Download file
                    self.logger.info(f"Downloading file: {item['name']}")
                    result = self.download_file(item['id'], current_path)
                    
                    if result['success'] and result['files']:
                        stats["files_downloaded"] += 1
                        stats["bytes_downloaded"] += int(result['files'].get('size', 0))
                        result['files']['download_status'] = 'success'
                        all_files_metadata.append(result['files'])
                        self.logger.info(f"Successfully downloaded: {item['name']}")
                    else:
                        success = False
                        item['download_status'] = 'failed'
                        item['error_message'] = result.get('error', 'Unknown error')
                        all_files_metadata.append(item)
                        error_msg = f"Failed to download {item['name']}: {result.get('error', 'Unknown error')}"
                        stats["errors"].append(error_msg)
                        self.logger.error(error_msg)
                        
            except Exception as e:
                success = False
                item['download_status'] = 'error'
                item['error_message'] = str(e)
                all_files_metadata.append(item)
                error_msg = f"Error processing {item['name']}: {str(e)}"
                stats["errors"].append(error_msg)
                self.logger.error(error_msg)
                
        self.logger.info(f"Finished processing folder at {current_path}. Success: {success}")
        return success

    def download_folder(self, 
                       folder_id: str, 
                       save_path: str, 
                       file_types: Optional[List[str]] = None,
                       max_depth: int = -1,
                       skip_existing: bool = False) -> Dict[str, Union[bool, Dict, List[Dict]]]:
        """
        Recursively download files from a Google Drive folder.
        
        Args:
            folder_id (str): The ID of the folder to download
            save_path (str): Path where files should be saved
            file_types (List[str], optional): List of file extensions to download
            max_depth (int, optional): Maximum folder depth to traverse (-1 for unlimited)
            skip_existing (bool, optional): Skip files that already exist locally
            
        Returns:
            Dict: Summary of the download operation including:
                - success: bool indicating overall success
                - stats: download statistics
                - files: list of metadata for all processed files
                - error: error message if failed
        """
        if not self.service:
            self.logger.error("Google Drive service not initialized")
            return {
                "success": False,
                "error": "Service not initialized",
                "stats": {},
                "files": []
            }

        stats = {
            "files_processed": 0,
            "files_downloaded": 0,
            "files_skipped": 0,
            "bytes_downloaded": 0,
            "errors": [],
            "start_time": datetime.now()
        }
        
        all_files_metadata = []

        try:
            os.makedirs(save_path, exist_ok=True)
            success = self._process_folder(
                folder_id, 
                save_path, 
                0,
                max_depth,
                file_types,
                skip_existing,
                stats,
                all_files_metadata
            )
            
            # Calculate duration and add to stats
            stats["end_time"] = datetime.now()
            stats["duration"] = str(stats["end_time"] - stats["start_time"])
            
            return {
                "success": success,
                "stats": stats,
                "files": all_files_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading folder: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stats": stats,
                "files": all_files_metadata
            }


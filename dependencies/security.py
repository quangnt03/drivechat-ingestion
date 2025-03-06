from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import boto3

reusable_oauth2 = HTTPBearer(
    scheme_name='Authorization'
)

def validate_token(http_authorization_credentials=Depends(reusable_oauth2)) -> str:
    """
    Decode JWT token to get username => return username
    """
    client = boto3.client('cognito-idp', region_name='us-east-1')
    try:
        user = client.get_user(
            AccessToken=http_authorization_credentials.credentials
        )        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )
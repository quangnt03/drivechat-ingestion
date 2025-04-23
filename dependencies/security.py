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
    except Exception as _:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
        )
    is_verified = user['UserAttributes'][1]['Value']
    if is_verified != 'true':
        raise HTTPException(
            status_code=403,
            detail="User is not verified",
        )
    return user


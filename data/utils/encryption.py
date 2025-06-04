import base64

def encrypt_data(data: bytes) -> bytes:
    return base64.b64encode(data)

def decrypt_data(token: bytes) -> bytes:
    return base64.b64decode(token)

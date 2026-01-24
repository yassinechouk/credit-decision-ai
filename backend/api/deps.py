from typing import Optional, Dict
from fastapi import Header, HTTPException, status


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = authorization.replace("Bearer", "").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    parts = token.split(":")
    if len(parts) < 4 or parts[0] != "token":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

    role = parts[1]
    user_id_raw = parts[2]
    try:
        user_id = int(user_id_raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token user") from exc

    return {"user_id": user_id, "role": role}

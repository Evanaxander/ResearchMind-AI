"""
Auth Router
------------
Endpoints for user registration, login, and role management.

POST /auth/register  → create a new user with a role
POST /auth/login     → get a JWT token
GET  /auth/me        → get current user info
GET  /auth/roles     → list all available roles
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

from app.services import auth_service

router  = APIRouter()
bearer  = HTTPBearer(auto_error=False)


# ── Request / response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    role:     str
    email:    Optional[str] = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    role:         str
    role_label:   str


# ── Dependency: get current user from token ───────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer)
) -> Optional[dict]:
    """
    FastAPI dependency — extracts and validates the JWT token.

    Usage in any endpoint:
        user = Depends(get_current_user)

    Returns the decoded token payload (username + role)
    or None if no token provided (for optional auth).
    """
    if not credentials:
        return None

    payload = auth_service.decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again."
        )
    return payload


def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer)
) -> dict:
    """
    Stricter version — raises 401 if no valid token.
    Use this for endpoints that require authentication.
    """
    user = get_current_user(credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. "
                   "Log in at POST /api/v1/auth/login to get a token."
        )
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/auth/register", status_code=201)
async def register(request: RegisterRequest):
    """
    Create a new FinanceIQ user.

    Roles available:
    - analyst          → full access, detailed answers
    - portfolio_manager → summaries + risk analysis
    - compliance       → contradiction reports + regulatory citations
    - executive        → brief summaries only

    In a real deployment, registration would be restricted to admins.
    """
    try:
        user = auth_service.create_user(
            username = request.username,
            password = request.password,
            role     = request.role,
            email    = request.email or "",
        )
        return {
            "message":  f"User '{request.username}' created successfully.",
            "username": user["username"],
            "role":     user["role"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Log in and receive a JWT token.

    Include the token in subsequent requests:
        Authorization: Bearer <token>

    Token expires after 24 hours.
    """
    user = auth_service.authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    token      = auth_service.create_token(user["username"], user["role"])
    role_info  = auth_service.ROLES.get(user["role"], {})

    return TokenResponse(
        access_token = token,
        username     = user["username"],
        role         = user["role"],
        role_label   = role_info.get("label", user["role"]),
    )


@router.get("/auth/me")
async def get_me(user: dict = Depends(require_user)):
    """
    Returns the current user's profile and role permissions.

    Shows exactly what this role can and cannot access —
    useful for the frontend to conditionally show UI elements.
    """
    role_config = auth_service.ROLES.get(user["role"], {})
    db_user     = auth_service.get_user(user["sub"])

    return {
        "username":    user["sub"],
        "role":        user["role"],
        "role_label":  role_config.get("label"),
        "permissions": {
            "answer_depth":   role_config.get("answer_depth"),
            "max_chunks":     role_config.get("max_chunks"),
            "show_raw_text":  role_config.get("show_raw_text"),
            "show_analysis":  role_config.get("show_analysis"),
            "can_query_contradictions": role_config.get(
                "can_query_contradictions"
            ),
        },
        "last_login":  db_user.get("last_login") if db_user else None,
    }


@router.get("/auth/roles")
async def list_roles():
    """
    Returns all available roles and their permissions.
    Used by registration UI to show role options.
    """
    return {
        "roles": [
            {
                "id":          role_id,
                "label":       config["label"],
                "description": config["description"],
                "answer_depth": config["answer_depth"],
            }
            for role_id, config in auth_service.ROLES.items()
        ]
    }
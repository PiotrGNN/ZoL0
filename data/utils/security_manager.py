"""Security management module."""

import os
import logging
import hmac
import hashlib
import json
import re
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import jwt
import bcrypt
import secrets
from data.logging.system_logger import get_logger

logger = get_logger()

class SecurityConfig:
    """Security configuration container."""
    def __init__(self,
                 jwt_secret: str = None,
                 jwt_expiry_hours: int = 24,
                 password_min_length: int = 8,
                 password_require_special: bool = True,
                 max_login_attempts: int = 5,
                 lockout_duration_minutes: int = 15,
                 api_rate_limit_per_minute: int = 60):
        self.jwt_secret = jwt_secret or os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        self.jwt_expiry_hours = jwt_expiry_hours
        self.password_min_length = password_min_length
        self.password_require_special = password_require_special
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.api_rate_limit_per_minute = api_rate_limit_per_minute

class SecurityManager:
    """Enhanced security management system."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.failed_attempts: Dict[str, list] = {}
        self.active_tokens: Dict[str, datetime] = {}
        self.api_calls: Dict[str, list] = {}

    def validate_password(self, password: str) -> tuple[bool, Optional[str]]:
        """Validate password strength."""
        if len(password) < self.config.password_min_length:
            return False, f"Password must be at least {self.config.password_min_length} characters"
        
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r"\d", password):
            return False, "Password must contain at least one digit"
        
        if self.config.password_require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        
        return True, None

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            logger.log_error(f"Error verifying password: {e}")
            return False

    def create_token(self, user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create JWT token."""
        try:
            claims = {
                'sub': user_id,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours)
            }
            
            if additional_claims:
                claims.update(additional_claims)
            
            token = jwt.encode(claims, self.config.jwt_secret, algorithm='HS256')
            self.active_tokens[token] = claims['exp']
            return token
        except Exception as e:
            logger.log_error(f"Error creating token: {e}")
            return None

    def verify_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token."""
        try:
            if token not in self.active_tokens:
                return False, None
            
            if datetime.utcnow() > self.active_tokens[token]:
                self.revoke_token(token)
                return False, None
            
            claims = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            return True, claims
        except jwt.ExpiredSignatureError:
            self.revoke_token(token)
            return False, None
        except Exception as e:
            logger.log_error(f"Error verifying token: {e}")
            return False, None

    def revoke_token(self, token: str) -> None:
        """Revoke JWT token."""
        self.active_tokens.pop(token, None)

    def check_rate_limit(self, api_key: str) -> bool:
        """Check API rate limit."""
        now = datetime.now()
        if api_key not in self.api_calls:
            self.api_calls[api_key] = []
        
        # Remove old calls
        self.api_calls[api_key] = [
            time for time in self.api_calls[api_key]
            if now - time < timedelta(minutes=1)
        ]
        
        # Check limit
        if len(self.api_calls[api_key]) >= self.config.api_rate_limit_per_minute:
            return False
        
        # Add new call
        self.api_calls[api_key].append(now)
        return True

    def record_failed_attempt(self, user_id: str) -> bool:
        """Record failed login attempt."""
        now = datetime.now()
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Remove old attempts
        self.failed_attempts[user_id] = [
            time for time in self.failed_attempts[user_id]
            if now - time < timedelta(minutes=self.config.lockout_duration_minutes)
        ]
        
        # Add new attempt
        self.failed_attempts[user_id].append(now)
        
        return len(self.failed_attempts[user_id]) >= self.config.max_login_attempts

    def reset_failed_attempts(self, user_id: str) -> None:
        """Reset failed login attempts."""
        self.failed_attempts.pop(user_id, None)

    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked."""
        if user_id not in self.failed_attempts:
            return False
        
        now = datetime.now()
        recent_attempts = [
            time for time in self.failed_attempts[user_id]
            if now - time < timedelta(minutes=self.config.lockout_duration_minutes)
        ]
        
        return len(recent_attempts) >= self.config.max_login_attempts

    def generate_api_credentials(self) -> tuple[str, str]:
        """Generate new API key and secret."""
        api_key = secrets.token_urlsafe(32)
        api_secret = secrets.token_urlsafe(64)
        return api_key, api_secret

    def verify_api_signature(self, api_key: str, api_secret: str, payload: str, signature: str) -> bool:
        """Verify API request signature."""
        try:
            expected = hmac.new(
                api_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except Exception as e:
            logger.log_error(f"Error verifying API signature: {e}")
            return False

    def sanitize_input(self, data: str) -> str:
        """Sanitize user input."""
        # Remove potential SQL injection patterns
        data = re.sub(r"['\"\\;()]", "", data)
        # Remove potential XSS patterns
        data = re.sub(r"[<>]", "", data)
        return data

    def validate_json(self, data: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JSON data."""
        try:
            parsed = json.loads(data)
            return True, parsed
        except json.JSONDecodeError:
            return False, None

def require_auth(f: Callable) -> Callable:
    """Decorator to require valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = kwargs.get('token')
        if not token:
            return {'error': 'No token provided'}, 401
        
        security_manager = get_security_manager()
        valid, claims = security_manager.verify_token(token)
        
        if not valid:
            return {'error': 'Invalid token'}, 401
        
        kwargs['user_claims'] = claims
        return f(*args, **kwargs)
    return decorated

def rate_limit(f: Callable) -> Callable:
    """Decorator to apply API rate limiting."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = kwargs.get('api_key')
        if not api_key:
            return {'error': 'No API key provided'}, 401
        
        security_manager = get_security_manager()
        if not security_manager.check_rate_limit(api_key):
            return {'error': 'Rate limit exceeded'}, 429
        
        return f(*args, **kwargs)
    return decorated

# Global security manager instance with default config
_security_manager = SecurityManager()

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    return _security_manager
"""
Moduł zarządzania bezpieczeństwem systemu z obsługą:
- Uwierzytelniania i autoryzacji
- Rate limitingu
- Walidacji danych wejściowych
- Monitorowania bezpieczeństwa
- Logowania zdarzeń bezpieczeństwa
"""

import os
import jwt
import json
import time
import logging
import hashlib
import bleach
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import redis
from redis.exceptions import RedisError

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        """Inicjalizacja menedżera bezpieczeństwa."""
        self.secret_key = os.getenv('JWT_SECRET_KEY', os.urandom(32))
        self.token_expiry = int(os.getenv('JWT_EXPIRY_HOURS', 24))
        
        # Konfiguracja Redis dla rate limitingu i cache
        try:
            self.redis = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )
            self.redis.ping()
            logger.info("Połączono z Redis")
        except RedisError as e:
            logger.warning(f"Nie można połączyć z Redis: {e}")
            self.redis = None
            
        # Limity zapytań dla różnych typów endpointów
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 zapytań na minutę
            'trading': {'requests': 20, 'window': 60},   # 20 zapytań na minutę
            'critical': {'requests': 5, 'window': 60}    # 5 zapytań na minutę
        }
        
        # Cache zablokowanych IP
        self.blocked_ips = set()
        self.ip_fail_counts = defaultdict(int)
        self.block_threshold = 5  # Liczba nieudanych prób przed blokadą
        self.block_duration = 3600  # Czas blokady w sekundach
        
        # Lista aktywnych tokenów
        self.active_tokens = set()
        self.revoked_tokens = set()
        
    def generate_token(self, payload: Dict[str, Any]) -> str:
        """
        Generuje token JWT z dodatkowymi zabezpieczeniami.
        
        Args:
            payload: Dane do zakodowania w tokenie
            
        Returns:
            str: Wygenerowany token JWT
        """
        # Dodaj standardowe pola
        now = datetime.utcnow()
        payload.update({
            'iat': now,
            'exp': now + timedelta(hours=self.token_expiry),
            'jti': hashlib.sha256(os.urandom(32)).hexdigest()
        })
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        self.active_tokens.add(token)
        
        # Zapisz token w Redis jeśli dostępne
        if self.redis:
            try:
                self.redis.setex(
                    f"token:{token}",
                    self.token_expiry * 3600,
                    json.dumps({
                        'user': payload.get('sub'),
                        'created_at': now.isoformat()
                    })
                )
            except RedisError as e:
                logger.error(f"Błąd podczas zapisywania tokenu w Redis: {e}")
                
        return token
        
    def verify_token(
        self,
        token: str,
        verify_ip: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Weryfikuje token JWT z dodatkowymi zabezpieczeniami.
        
        Args:
            token: Token JWT do weryfikacji
            verify_ip: Opcjonalnie IP do weryfikacji
            
        Returns:
            Tuple[bool, Dict]: (Czy token jest ważny, Payload/komunikat błędu)
        """
        if token in self.revoked_tokens:
            return False, {'error': 'Token został unieważniony'}
            
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Sprawdź czy token jest aktywny
            if token not in self.active_tokens:
                if self.redis:
                    try:
                        if not self.redis.exists(f"token:{token}"):
                            return False, {'error': 'Token nie jest aktywny'}
                    except RedisError as e:
                        logger.error(f"Błąd podczas sprawdzania tokenu w Redis: {e}")
                else:
                    return False, {'error': 'Token nie jest aktywny'}
            
            # Sprawdź IP jeśli podane
            if verify_ip and payload.get('ip') != verify_ip:
                self.log_security_event('ip_mismatch', {
                    'token_ip': payload.get('ip'),
                    'request_ip': verify_ip
                })
                return False, {'error': 'Niedozwolone IP'}
                
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token wygasł'}
        except jwt.InvalidTokenError as e:
            return False, {'error': f'Nieprawidłowy token: {str(e)}'}
            
    def revoke_token(self, token: str) -> bool:
        """
        Unieważnia token JWT.
        
        Args:
            token: Token do unieważnienia
            
        Returns:
            bool: Czy operacja się powiodła
        """
        try:
            self.active_tokens.discard(token)
            self.revoked_tokens.add(token)
            
            if self.redis:
                try:
                    self.redis.delete(f"token:{token}")
                except RedisError as e:
                    logger.error(f"Błąd podczas usuwania tokenu z Redis: {e}")
                    
            return True
        except Exception as e:
            logger.error(f"Błąd podczas unieważniania tokenu: {e}")
            return False
            
    def check_rate_limit(self, ip: str, endpoint_type: str = 'default') -> bool:
        """
        Sprawdza limit zapytań dla danego IP.
        
        Args:
            ip: Adres IP do sprawdzenia
            endpoint_type: Typ endpointu (default/trading/critical)
            
        Returns:
            bool: Czy zapytanie mieści się w limicie
        """
        if ip in self.blocked_ips:
            return False
            
        if self.redis:
            try:
                key = f"rate_limit:{ip}:{endpoint_type}"
                current = int(self.redis.get(key) or 0)
                
                if current >= self.rate_limits[endpoint_type]['requests']:
                    return False
                    
                pipe = self.redis.pipeline()
                pipe.incr(key)
                pipe.expire(key, self.rate_limits[endpoint_type]['window'])
                pipe.execute()
                
                return True
            except RedisError as e:
                logger.error(f"Błąd podczas sprawdzania rate limit w Redis: {e}")
                return True  # Failsafe
        
        # Prosty rate limiting w pamięci jeśli Redis niedostępny
        current_time = time.time()
        key = f"{ip}:{endpoint_type}"
        
        if not hasattr(self, '_rate_limit_memory'):
            self._rate_limit_memory = {}
            
        if key in self._rate_limit_memory:
            requests, window_start = self._rate_limit_memory[key]
            if current_time - window_start >= self.rate_limits[endpoint_type]['window']:
                self._rate_limit_memory[key] = (1, current_time)
                return True
            elif requests >= self.rate_limits[endpoint_type]['requests']:
                return False
            else:
                self._rate_limit_memory[key] = (requests + 1, window_start)
                return True
        else:
            self._rate_limit_memory[key] = (1, current_time)
            return True
            
    def is_ip_blocked(self, ip: str) -> bool:
        """
        Sprawdza czy IP jest zablokowane.
        
        Args:
            ip: Adres IP do sprawdzenia
            
        Returns:
            bool: Czy IP jest zablokowane
        """
        if self.redis:
            try:
                return bool(self.redis.exists(f"blocked_ip:{ip}"))
            except RedisError as e:
                logger.error(f"Błąd podczas sprawdzania blokady IP w Redis: {e}")
                return ip in self.blocked_ips
        return ip in self.blocked_ips
        
    def block_ip(self, ip: str, duration: int = None) -> None:
        """
        Blokuje IP na określony czas.
        
        Args:
            ip: Adres IP do zablokowania
            duration: Czas blokady w sekundach (domyślnie self.block_duration)
        """
        duration = duration or self.block_duration
        self.blocked_ips.add(ip)
        
        if self.redis:
            try:
                self.redis.setex(f"blocked_ip:{ip}", duration, 1)
            except RedisError as e:
                logger.error(f"Błąd podczas blokowania IP w Redis: {e}")
                
        self.log_security_event('ip_blocked', {
            'ip': ip,
            'duration': duration
        })
        
    def unblock_ip(self, ip: str) -> None:
        """Odblokowuje IP."""
        self.blocked_ips.discard(ip)
        
        if self.redis:
            try:
                self.redis.delete(f"blocked_ip:{ip}")
            except RedisError as e:
                logger.error(f"Błąd podczas odblokowywania IP w Redis: {e}")
                
        self.log_security_event('ip_unblocked', {
            'ip': ip
        })
        
    def record_failed_attempt(self, ip: str) -> None:
        """
        Zapisuje nieudaną próbę logowania i blokuje IP po przekroczeniu progu.
        
        Args:
            ip: Adres IP do zapisania
        """
        self.ip_fail_counts[ip] += 1
        
        if self.ip_fail_counts[ip] >= self.block_threshold:
            self.block_ip(ip)
            self.log_security_event('ip_blocked_attempts', {
                'ip': ip,
                'attempts': self.ip_fail_counts[ip]
            })
            
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Sprawdza siłę hasła.
        
        Args:
            password: Hasło do sprawdzenia
            
        Returns:
            Tuple[bool, str]: (Czy hasło jest wystarczająco silne, komunikat)
        """
        if len(password) < 8:
            return False, "Hasło musi mieć co najmniej 8 znaków"
            
        if not any(c.isupper() for c in password):
            return False, "Hasło musi zawierać wielką literę"
            
        if not any(c.islower() for c in password):
            return False, "Hasło musi zawierać małą literę"
            
        if not any(c.isdigit() for c in password):
            return False, "Hasło musi zawierać cyfrę"
            
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Hasło musi zawierać znak specjalny"
            
        return True, "Hasło spełnia wymagania"
        
    def hash_password(self, password: str) -> Tuple[str, str]:
        """
        Haszuje hasło z użyciem soli.
        
        Args:
            password: Hasło do zahaszowania
            
        Returns:
            Tuple[str, str]: (Hash hasła, sól)
        """
        salt = os.urandom(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # Liczba iteracji
        )
        return password_hash.hex(), salt.hex()
        
    def verify_password(
        self,
        password: str,
        stored_hash: str,
        salt: str
    ) -> bool:
        """
        Weryfikuje hasło z hashem.
        
        Args:
            password: Hasło do sprawdzenia
            stored_hash: Zapisany hash hasła
            salt: Sól użyta do haszowania
            
        Returns:
            bool: Czy hasło jest prawidłowe
        """
        salt = bytes.fromhex(salt)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        ).hex()
        return password_hash == stored_hash
        
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Czyści dane wejściowe z potencjalnie niebezpiecznych elementów.
        
        Args:
            data: Dane do wyczyszczenia
            
        Returns:
            Dict: Wyczyszczone dane
        """
        if isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(v) for v in data]
        elif isinstance(data, str):
            return bleach.clean(data)
        else:
            return data
            
    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Loguje zdarzenie bezpieczeństwa.
        
        Args:
            event_type: Typ zdarzenia
            details: Szczegóły zdarzenia
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        }
        
        logger.info(f"Zdarzenie bezpieczeństwa: {json.dumps(event)}")
        
        if self.redis:
            try:
                self.redis.lpush('security_events', json.dumps(event))
                self.redis.ltrim('security_events', 0, 999)  # Zachowaj ostatnie 1000 zdarzeń
            except RedisError as e:
                logger.error(f"Błąd podczas zapisywania zdarzenia w Redis: {e}")
                
    def get_security_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Pobiera historię zdarzeń bezpieczeństwa.
        
        Args:
            event_type: Opcjonalny filtr typu zdarzeń
            limit: Maksymalna liczba zdarzeń do pobrania
            
        Returns:
            List[Dict]: Lista zdarzeń bezpieczeństwa
        """
        events = []
        
        if self.redis:
            try:
                all_events = self.redis.lrange('security_events', 0, limit - 1)
                events = [json.loads(event) for event in all_events]
            except RedisError as e:
                logger.error(f"Błąd podczas pobierania zdarzeń z Redis: {e}")
                
        if event_type:
            events = [e for e in events if e['type'] == event_type]
            
        return events[:limit]

    def cleanup_expired_sessions(self) -> None:
        """Czyści wygasłe sesje i tokeny."""
        current_time = datetime.utcnow()
        
        # Wyczyść wygasłe tokeny z pamięci
        expired_tokens = set()
        for token in self.active_tokens:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                exp_time = datetime.fromtimestamp(payload['exp'])
                if exp_time <= current_time:
                    expired_tokens.add(token)
            except jwt.InvalidTokenError:
                expired_tokens.add(token)
                
        self.active_tokens -= expired_tokens
        
        # Wyczyść zablokowane IP po czasie
        if self.redis:
            try:
                # Redis automatycznie usuwa wygasłe klucze
                pass
            except RedisError as e:
                logger.error(f"Błąd podczas czyszczenia sesji w Redis: {e}")
        else:
            # Czyszczenie z pamięci
            for ip in list(self.blocked_ips):
                if time.time() - self._rate_limit_memory.get(f"{ip}:block_time", 0) > self.block_duration:
                    self.unblock_ip(ip)

# Instancja singletona
security_manager = SecurityManager()
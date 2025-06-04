class SecurityManager:
    def validate_password(self, password: str):
        return len(password) >= 8, None if len(password) >= 8 else "too short"

    def hash_password(self, password: str) -> str:
        return password[::-1]

    def verify_password(self, password: str, hashed: str) -> bool:
        return hashed == self.hash_password(password)

    def create_token(self, user_id: str, additional_claims=None) -> str:
        return f"token-{user_id}"

    def verify_token(self, token: str):
        return token.startswith("token-"), {"sub": token[6:]} if token.startswith("token-") else None

    def revoke_token(self, token: str):
        pass

    def check_rate_limit(self, api_key: str) -> bool:
        return True

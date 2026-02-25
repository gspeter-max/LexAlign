from huggingface_hub import whoami
from huggingface_hub.utils import HfHubHTTPError


class AuthError(Exception):
    """Raised when authentication fails."""


class AuthManager:
    """Manage Hugging Face authentication."""

    def __init__(self, token: str):
        """
        Initialize authentication manager.

        Args:
            token: Hugging Face API token

        Raises:
            AuthError: If token is empty
        """
        if not token or not token.strip():
            raise AuthError("Token cannot be empty")
        self.token = token.strip()

    def is_authenticated(self) -> bool:
        """Check if a token is configured."""
        return bool(self.token)

    def validate_token(self, raise_on_error: bool = False) -> bool:
        """
        Validate the token by calling Hugging Face API.

        Args:
            raise_on_error: If True, raise AuthError on validation failure

        Returns:
            True if token is valid, False otherwise

        Raises:
            AuthError: If validation fails and raise_on_error is True
        """
        try:
            whoami(token=self.token)
            return True
        except HfHubHTTPError as e:
            if raise_on_error:
                raise AuthError(f"Authentication failed: {e}")
            return False
        except Exception as e:
            if raise_on_error:
                raise AuthError(f"Authentication error: {e}")
            return False

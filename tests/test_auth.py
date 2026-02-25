import pytest
from lexalign.downloader.auth import AuthManager, AuthError
from unittest.mock import MagicMock
import requests

def test_auth_with_valid_token():
    auth = AuthManager("valid_token_123")
    assert auth.token == "valid_token_123"
    assert auth.is_authenticated()

def test_auth_with_empty_token():
    with pytest.raises(AuthError, match="Token cannot be empty"):
        AuthManager("")

def test_validate_token_success(mocker):
    auth = AuthManager("test_token")
    mock_whoami = mocker.patch('lexalign.downloader.auth.whoami', return_value={"type": "user"})

    result = auth.validate_token()
    assert result is True
    mock_whoami.assert_called_once_with(token="test_token")

def test_validate_token_invalid(mocker):
    auth = AuthManager("invalid_token")
    mock_response = MagicMock()
    mock_response.status_code = 401
    http_error = requests.exceptions.HTTPError("Unauthorized", response=mock_response)

    mock_whoami = mocker.patch(
        'lexalign.downloader.auth.whoami',
        side_effect=http_error
    )

    result = auth.validate_token()
    assert result is False

def test_validate_token_required(mocker):
    auth = AuthManager("test_token")
    mock_response = MagicMock()
    mock_response.status_code = 401
    http_error = requests.exceptions.HTTPError("401 Client Error", response=mock_response)

    mock_whoami = mocker.patch(
        'lexalign.downloader.auth.whoami',
        side_effect=http_error
    )

    with pytest.raises(AuthError, match="Authentication (failed|error)"):
        auth.validate_token(raise_on_error=True)

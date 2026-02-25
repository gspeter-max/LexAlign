# lexalign/utils/device.py
"""Device detection and selection utilities."""

import torch
from typing import Tuple


class DeviceError(Exception):
    """Device-related errors."""


class DeviceManager:
    """Manages device detection and selection."""

    VALID_DEVICES = ("cuda", "cpu")

    def detect_device(self) -> Tuple[str, bool]:
        """
        Auto-detect the best available device.

        Returns:
            Tuple of (device_string, fell_back_from_cuda).
            fell_back_from_cuda is always False for auto-detection.
        """
        if torch.cuda.is_available():
            return "cuda", False
        return "cpu", False

    def get_device(self, requested: str = None) -> Tuple[str, bool]:
        """
        Get device string with automatic fallback logic.

        Args:
            requested: Requested device ("cuda" or "cpu"), or None for auto.

        Returns:
            Tuple of (device_string, fell_back_from_cuda).
            fell_back_from_cuda is True only when "cuda" was requested
            but CUDA is unavailable and we fell back to "cpu".

        Raises:
            DeviceError: If an invalid device name is provided
        """
        if requested is None:
            return self.detect_device()

        if requested not in self.VALID_DEVICES:
            raise DeviceError(
                f"Invalid device: {requested!r}. "
                f"Valid choices: {self.VALID_DEVICES}"
            )

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu", True  # Fallback with warning flag

        return requested, False

import torch


class DeviceError(Exception):
    """Device-related errors."""
    pass


class DeviceManager:
    """Manages device detection and selection."""

    def detect_device(self) -> tuple[str, bool]:
        """
        Auto-detect available device.

        Returns:
            Tuple of (device_string, fell_back_from_cuda)
        """
        if torch.cuda.is_available():
            return "cuda", False
        return "cpu", False

    def get_device(self, requested: str = None) -> tuple[str, bool]:
        """
        Get device with fallback logic.

        Args:
            requested: Requested device ("cuda" or "cpu")

        Returns:
            Tuple of (device_string, fell_back_from_cuda)

        Raises:
            DeviceError: If invalid device name
        """
        if requested is None:
            return self.detect_device(), False

        if requested not in ("cuda", "cpu"):
            raise DeviceError(f"Invalid device: {requested}. Use 'cuda' or 'cpu'.")

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu", True  # Fallback with warning flag

        return requested, False

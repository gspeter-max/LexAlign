import torch


class DeviceError(Exception):
    """Device-related errors."""
    pass


class DeviceManager:
    """Manages device detection and selection."""

    def detect_device(self) -> str:
        """
        Auto-detect available device.

        Returns:
            "cuda" if available, else "cpu"
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_device(self, requested: str = None) -> str:
        """
        Get device with fallback logic.

        Args:
            requested: Requested device ("cuda" or "cpu")

        Returns:
            Device string to use

        Raises:
            DeviceError: If invalid device name
        """
        if requested is None:
            return self.detect_device()

        if requested not in ("cuda", "cpu"):
            raise DeviceError(f"Invalid device: {requested}. Use 'cuda' or 'cpu'.")

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"  # Fallback

        return requested

from pathlib import Path
from typing import Optional
import re


class CheckpointManager:
    """Manage training checkpoints."""

    def find_latest(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find latest checkpoint directory.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Path to latest checkpoint, or None if no checkpoints
        """
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists():
            return None

        checkpoints = sorted(
            [p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1])
        )

        if not checkpoints:
            return None

        return str(checkpoints[-1])

    def get_step(self, checkpoint_path: str) -> int:
        """
        Extract step number from checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Step number
        """
        match = re.search(r"checkpoint-(\d+)", checkpoint_path)
        if match:
            return int(match.group(1))
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")

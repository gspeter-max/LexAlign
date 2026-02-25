# lexalign/finetuner/lora_config.py
"""LoRA and QLoRA configuration builder for LexAlign."""

import platform
import warnings
from typing import Any, Dict, Optional

from peft import LoraConfig, TaskType


class LoraConfigBuilder:
    """Build LoRA/QLoRA PEFT configurations."""

    def build(
        self,
        training_params: Dict[str, Any],
        quantization_bits: Optional[int] = None,  # reserved for future use
    ) -> LoraConfig:
        """
        Build a LoRA configuration from training params.

        Args:
            training_params: Training configuration dictionary
            quantization_bits: Reserved — quantization is handled separately
                via :meth:`get_quantization_config`.

        Returns:
            Configured :class:`peft.LoraConfig` instance
        """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_params.get("lora_r", 16),
            lora_alpha=training_params.get("lora_alpha", 32),
            lora_dropout=training_params.get("lora_dropout", 0.05),
            target_modules=training_params.get("target_modules"),
            bias="none",
        )

    def get_quantization_config(self, bits: int) -> Dict[str, bool]:
        """
        Get quantization configuration dict for QLoRA.

        Args:
            bits: Quantization bits — must be 4 or 8

        Returns:
            Dict suitable for unpacking into ``from_pretrained`` kwargs

        Raises:
            ValueError: If bits is not 4 or 8
        """
        if platform.system() != "Linux":
            warnings.warn(
                f"QLoRA (bitsandbytes) has limited support on {platform.system()}. "
                "For reliable QLoRA quantization, use a Linux machine with a CUDA GPU. "
                "See: https://github.com/TimDettmers/bitsandbytes#tldr",
                RuntimeWarning,
                stacklevel=2,
            )
        if bits == 4:
            return {"load_in_4bit": True}
        if bits == 8:
            return {"load_in_8bit": True}
        raise ValueError(f"Quantization must be 4 or 8, got {bits}")

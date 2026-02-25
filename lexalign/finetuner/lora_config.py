from peft import LoraConfig, get_peft_model, TaskType


class LoraConfigBuilder:
    """Build LoRA/QLoRA configurations."""

    def build(self, training_params: dict, quantization_bits: int = None):
        """
        Build LoRA configuration.

        Args:
            training_params: Training configuration dictionary
            quantization_bits: Quantization bits (4 or 8) for QLoRA

        Returns:
            LoraConfig instance
        """
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_params.get("lora_r", 16),
            lora_alpha=training_params.get("lora_alpha", 32),
            lora_dropout=training_params.get("lora_dropout", 0.05),
            target_modules=training_params.get("target_modules"),
            bias="none",
        )

    def get_quantization_config(self, bits: int) -> dict:
        """
        Get quantization configuration for QLoRA.

        Args:
            bits: Quantization bits (4 or 8)

        Returns:
            Dictionary with quantization settings

        Raises:
            ValueError: If bits is not 4 or 8
        """
        if bits == 4:
            return {"load_in_4bit": True}
        elif bits == 8:
            return {"load_in_8bit": True}
        else:
            raise ValueError(f"Quantization must be 4 or 8, got {bits}")

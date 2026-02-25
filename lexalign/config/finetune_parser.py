import yaml
from typing import Dict, Any


class ConfigError(Exception):
    """Configuration errors."""
    pass


class FinetuneConfigParser:
    """Parse and validate fine-tuning configuration."""

    DEFAULTS = {
        "dataset": {
            "format": "auto",
            "text_field": "text",
            "train_split": "train",
        },
        "training": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 3e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "save_steps": 500,
            "max_steps": None,
            "max_seq_length": 512,
            "packing": False,
        },
    }

    REQUIRED_FIELDS = ["model", "dataset", "training"]

    def parse(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML configuration with validation.

        Args:
            yaml_content: YAML configuration string

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigError: If validation fails
        """
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}")

        self._validate_required_fields(config)
        self._apply_defaults(config)
        self._validate_training_params(config)

        return config

    def _validate_required_fields(self, config: Dict[str, Any]):
        """Ensure all required top-level fields exist."""
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                raise ConfigError(f"Missing required field: {field}")

    def _apply_defaults(self, config: Dict[str, Any]):
        """Apply default values for optional fields."""
        for section, defaults in self.DEFAULTS.items():
            if section not in config:
                config[section] = {}
            for key, value in defaults.items():
                if key not in config[section]:
                    config[section][key] = value

    def _validate_training_params(self, config: Dict[str, Any]):
        """Validate training-specific parameters."""
        training = config["training"]
        method = training.get("method", "lora")

        if method not in ("lora", "qlora"):
            raise ConfigError(f"Invalid training method: {method}. Use 'lora' or 'qlora'.")

        if method == "qlora":
            bits = training.get("quantization_bits", 4)
            if bits not in (4, 8):
                raise ConfigError(f"Quantization must be 4 or 8, got {bits}")

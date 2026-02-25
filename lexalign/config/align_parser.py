# lexalign/config/align_parser.py
"""Alignment configuration parser for LexAlign."""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from lexalign.config.errors import ConfigError
from lexalign.config.base_parser import BaseConfigParser


class AlignConfigParser(BaseConfigParser):
    """Parse and validate alignment configuration."""

    DEFAULTS: Dict[str, Any] = {
        "dataset": {
            "format": "auto",
            "prompt_field": "prompt",
            "chosen_field": "chosen",
            "rejected_field": "rejected",
            "train_split": "train",
        },
        "alignment": {
            "method": "dpo",
            "beta": 0.1,
            "loss_type": "sigmoid",
            "group_delay_size": 4,
            "group_delay_weight": 0.5,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 1e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "save_steps": 500,
            "max_steps": None,
        },
    }

    REQUIRED_FIELDS = ["model", "dataset", "alignment"]

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
        self._generate_output_dir(config)
        self._validate_alignment_params(config)

        return config

    def _generate_output_dir(self, config: Dict[str, Any]) -> None:
        """Generate default output_dir if not specified."""
        if "output_dir" not in config["alignment"]:
            model_name = Path(config["model"]["path"]).name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config["alignment"]["output_dir"] = (
                f"./checkpoints/{model_name}-aligned-{timestamp}"
            )

    def _validate_alignment_params(self, config: Dict[str, Any]) -> None:
        """Validate alignment-specific parameters."""
        alignment = config["alignment"]
        method = alignment.get("method", "dpo")

        if method not in ("dpo", "gdpo"):
            raise ConfigError(
                f"Invalid alignment method: {method}. Use 'dpo' or 'gdpo'."
            )

        if method == "gdpo":
            group_delay_size = alignment.get("group_delay_size", 4)
            if group_delay_size < 2:
                raise ConfigError(
                    f"group_delay_size must be >= 2, got {group_delay_size}"
                )

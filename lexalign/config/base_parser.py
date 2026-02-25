# lexalign/config/base_parser.py
"""Shared base config parser for LexAlign."""

from typing import Dict, Any

from lexalign.config.errors import ConfigError


class BaseConfigParser:
    """
    Abstract base class for LexAlign YAML config parsers.

    Subclasses must define:
        - REQUIRED_FIELDS: list[str]
        - DEFAULTS: dict[str, dict]
    """

    REQUIRED_FIELDS: list = []
    DEFAULTS: Dict[str, Any] = {}

    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """Ensure all required top-level fields exist in the config.

        Args:
            config: Parsed configuration dictionary

        Raises:
            ConfigError: If a required field is missing
        """
        if not isinstance(config, dict):
            raise ConfigError("Config must be a YAML mapping (dictionary)")
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                raise ConfigError(f"Missing required field: {field}")

    def _apply_defaults(self, config: Dict[str, Any]) -> None:
        """Apply default values for optional fields defined in DEFAULTS.

        Args:
            config: Parsed configuration dictionary (mutated in-place)
        """
        for section, defaults in self.DEFAULTS.items():
            if section not in config:
                config[section] = {}
            for key, value in defaults.items():
                if key not in config[section]:
                    config[section][key] = value

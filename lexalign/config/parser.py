import os
import re
from typing import Dict, Any, List

import yaml


class ConfigValidationError(Exception):
    """Raised when config validation fails."""


class ConfigParser:
    """Parse and validate YAML configuration files."""

    def _expand_env_vars(self, value: str, env: Dict[str, str]) -> str:
        """Expand environment variables in string values."""
        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}'

        def replace_env(match):
            var_name = match.group(1)
            if var_name not in env:
                raise ConfigValidationError(
                    f"Environment variable '{var_name}' not found"
                )
            return env[var_name]

        return re.sub(pattern, replace_env, value)

    def _validate_repo_config(self, repo: Dict[str, Any], repo_type: str) -> None:
        """Validate a single repo configuration."""
        required_fields = ["repo", "files", "output_dir"]
        for field in required_fields:
            if field not in repo:
                raise ConfigValidationError(
                    f"{repo_type} missing required field: {field}"
                )

        if not isinstance(repo["files"], list) or len(repo["files"]) == 0:
            raise ConfigValidationError(
                f"{repo_type} must have at least one file pattern"
            )

        if not repo["repo"]:
            raise ConfigValidationError(f"{repo_type} repo cannot be empty")

    def parse(self, yaml_content: str, env: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Parse and validate YAML configuration.

        Args:
            yaml_content: Raw YAML string
            env: Environment variables for expansion (defaults to os.environ)

        Returns:
            Parsed and validated configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        if env is None:
            env = dict(os.environ)

        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML: {e}")

        if not isinstance(config, dict):
            raise ConfigValidationError("Config must be a dictionary")

        # Expand environment variables in token
        if "huggingface" in config and "token" in config["huggingface"]:
            token = config["huggingface"]["token"]
            if isinstance(token, str):
                config["huggingface"]["token"] = self._expand_env_vars(token, env)

        # Validate that at least models or datasets are present
        has_models = "models" in config and config["models"]
        has_datasets = "datasets" in config and config["datasets"]

        if not has_models and not has_datasets:
            raise ConfigValidationError(
                "Config must have models or datasets section"
            )

        # Validate model configs
        if has_models:
            if not isinstance(config["models"], list):
                raise ConfigValidationError("models must be a list")
            for model in config["models"]:
                self._validate_repo_config(model, "model")

        # Validate dataset configs
        if has_datasets:
            if not isinstance(config["datasets"], list):
                raise ConfigValidationError("datasets must be a list")
            for dataset in config["datasets"]:
                self._validate_repo_config(dataset, "dataset")

        return config

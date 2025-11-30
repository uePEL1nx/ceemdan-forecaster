# -*- coding: utf-8 -*-
"""Configuration manager for parameters.yml."""
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigManager:
    """Manages reading/writing conf/base/parameters.yml."""

    def __init__(self, config_path: Path = None):
        """Initialize ConfigManager.

        Args:
            config_path: Path to parameters.yml. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "conf" / "base" / "parameters.yml"
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        """Load current parameters from YAML.

        Returns:
            Dictionary containing all parameters.
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def save(self, params: Dict[str, Any]) -> None:
        """Save parameters to YAML (overwrites base config).

        Args:
            params: Dictionary of parameters to save.
        """
        # Preserve header comments
        header = '''# ============================================================
# CEEMDAN-Informer-LSTM Pipeline Parameters
# Based on Li et al. (2024) methodology
# ============================================================

'''
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(header)
            yaml.dump(params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update a specific section of parameters.

        Args:
            section: Name of the section to update (e.g., 'informer', 'lstm').
            values: Dictionary of values to update within that section.
        """
        params = self.load()
        if section in params:
            params[section].update(values)
        else:
            params[section] = values
        self.save(params)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific section of parameters.

        Args:
            section: Name of the section to get.

        Returns:
            Dictionary containing the section's parameters.
        """
        params = self.load()
        return params.get(section, {})

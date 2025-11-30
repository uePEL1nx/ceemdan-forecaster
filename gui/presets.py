# -*- coding: utf-8 -*-
"""Preset configurations for quick setup.

Presets allow users to quickly switch between different training configurations:
- quick_test: Fast iteration for development (~15 minutes)
- standard: Balanced speed/quality (~1 hour)
- production: Full paper methodology (~2.5 hours)
"""
from typing import Dict, Any


# Production preset - Full paper methodology (default)
PRODUCTION = {
    "ensemble": {"n_models": 50},
    "ceemdan": {"trials": 100, "epsilon": 0.005},
    "informer": {"epochs": 10, "patience": 3},
    "lstm": {"epochs": 100, "patience": 10},
}

# Standard preset - Balanced speed/quality
STANDARD = {
    "ensemble": {"n_models": 20},
    "ceemdan": {"trials": 50, "epsilon": 0.005},
    "informer": {"epochs": 7, "patience": 3},
    "lstm": {"epochs": 50, "patience": 7},
}

# Quick test preset - Fast iteration
QUICK_TEST = {
    "ensemble": {"n_models": 5},
    "ceemdan": {"trials": 20, "epsilon": 0.005},
    "informer": {"epochs": 3, "patience": 2},
    "lstm": {"epochs": 20, "patience": 5},
}

# All presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "production": PRODUCTION,
    "standard": STANDARD,
    "quick_test": QUICK_TEST,
}

# Preset descriptions for UI
PRESET_DESCRIPTIONS = {
    "production": "Full paper methodology - ~2.5 hours, 450 models",
    "standard": "Balanced speed/quality - ~1 hour, 180 models",
    "quick_test": "Fast iteration - ~15 minutes, 45 models",
}


def apply_preset(current_config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Apply a preset to the current configuration.

    Args:
        current_config: Current configuration dictionary.
        preset_name: Name of the preset to apply ('production', 'standard', 'quick_test').

    Returns:
        Updated configuration dictionary with preset values applied.

    Raises:
        ValueError: If preset_name is not recognized.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    # Deep copy to avoid modifying the original
    import copy
    config = copy.deepcopy(current_config)

    preset = PRESETS[preset_name]
    for section, values in preset.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values

    return config


def get_preset_info(preset_name: str) -> str:
    """Get description for a preset.

    Args:
        preset_name: Name of the preset.

    Returns:
        Description string for the preset.
    """
    return PRESET_DESCRIPTIONS.get(preset_name, f"Unknown preset: {preset_name}")

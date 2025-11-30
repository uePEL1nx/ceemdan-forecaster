# -*- coding: utf-8 -*-
"""Custom Kedro dataset for PyTorch model checkpoints."""
from pathlib import Path
from typing import Any, Dict

import torch
from kedro.io import AbstractDataset


class PyTorchCheckpointDataset(AbstractDataset):
    """Dataset for saving/loading PyTorch model checkpoints.

    Example catalog entry:
    ```yaml
    my_model_checkpoint:
      type: ceemdan_forecaster.datasets.PyTorchCheckpointDataset
      filepath: data/06_models/my_model.pth
      device: auto
    ```
    """

    def __init__(self, filepath: str, device: str = "auto"):
        """Initialize the dataset.

        Args:
            filepath: Path to the checkpoint file
            device: Device to load the checkpoint onto.
                    'auto' will use CUDA if available.
        """
        self._filepath = Path(filepath)
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load(self) -> Dict[str, Any]:
        """Load a PyTorch checkpoint.

        Returns:
            Dictionary containing model state_dict and metadata.
        """
        return torch.load(
            self._filepath,
            map_location=self._device,
            weights_only=False
        )

    def _save(self, data: Dict[str, Any]) -> None:
        """Save a PyTorch checkpoint.

        Args:
            data: Dictionary containing model state_dict and metadata.
        """
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, self._filepath)

    def _exists(self) -> bool:
        """Check if the checkpoint file exists.

        Returns:
            True if the file exists, False otherwise.
        """
        return self._filepath.exists()

    def _describe(self) -> Dict[str, Any]:
        """Return a description of the dataset.

        Returns:
            Dictionary with filepath and device information.
        """
        return {
            "filepath": str(self._filepath),
            "device": self._device
        }

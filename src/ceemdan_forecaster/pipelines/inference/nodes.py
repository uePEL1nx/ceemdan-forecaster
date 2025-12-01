# -*- coding: utf-8 -*-
"""Ensemble inference nodes.

This module implements inference using trained ensemble models:
1. Load trained models (Informer for H-IMF, LSTM for L-IMF)
2. Generate predictions for each component
3. Average predictions across seeds (ensemble)
4. Denormalize predictions
5. Sum components + residual for final forecast
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Add LTSM project to path for model imports
LTSM_ROOT = Path(__file__).parents[5]
SRC_PATH = LTSM_ROOT / "src"
INFORMER_PATH = LTSM_ROOT / "Informer2020"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(INFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(INFORMER_PATH))


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class InformerDataset(Dataset):
    """Dataset for Informer model inference.

    CRITICAL: During inference, we must NOT pass the actual future values
    to the decoder. The decoder input (seq_y) should have:
    - label_len known historical values (for teacher forcing context)
    - pred_len ZEROS (masked future values - model must predict these)

    This prevents look-ahead bias where the model could "cheat" by
    seeing the actual values it's supposed to predict.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 1,
    ):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.valid_len = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.valid_len)

    def __getitem__(self, idx):
        seq_x = self.data[idx : idx + self.seq_len]

        # Decoder input: label_len known values + pred_len ZEROS
        # This prevents look-ahead bias during inference
        s_begin = idx + self.seq_len - self.label_len
        s_end = idx + self.seq_len  # Only up to known data, NOT including future
        label_part = self.data[s_begin:s_end]  # label_len known values

        # Mask future values with zeros - model must predict these
        seq_y = np.concatenate([label_part, np.zeros(self.pred_len)])

        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        seq_x = torch.FloatTensor(seq_x).reshape(-1, 1)
        seq_y = torch.FloatTensor(seq_y).reshape(-1, 1)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class LSTMDataset(Dataset):
    """Dataset for LSTM model inference."""

    def __init__(self, data: np.ndarray, look_back: int = 20, pred_len: int = 1):
        self.data = data
        self.look_back = look_back
        self.pred_len = pred_len
        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.look_back - self.pred_len + 1):
            X.append(self.data[i : i + self.look_back])
            y.append(self.data[i + self.look_back : i + self.look_back + self.pred_len])
        return np.array(X) if X else np.array([]).reshape(0, self.look_back), \
               np.array(y) if y else np.array([]).reshape(0, self.pred_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).reshape(-1, 1)
        y = torch.FloatTensor(self.y[idx]).reshape(-1, 1)
        return x, y


class LSTMModel(nn.Module):
    """LSTM model for L-IMF prediction."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 4,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.1,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        x = x.to(self.device)
        if hidden is None:
            batch_size = x.size(0)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            hidden = (h_0, c_0)

        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output, hidden


def create_informer_model(params: Dict[str, Any], device: torch.device):
    """Create Informer model with given parameters."""
    from models.model import Informer

    model = Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=params.get("seq_len", 96),
        label_len=params.get("label_len", 48),
        out_len=params.get("pred_len", 1),
        factor=5,
        d_model=params.get("d_model", 256),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 2),
        d_layers=params.get("d_layers", 1),
        d_ff=params.get("d_ff", 512),
        dropout=params.get("dropout", 0.05),
        attn=params.get("attn", "prob"),
        embed=params.get("embed", "fixed"),
        freq="d",
        activation=params.get("activation", "gelu"),
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    )
    return model.to(device)


def predict_with_informer(
    model: nn.Module,
    data: np.ndarray,
    params: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Generate predictions with an Informer model."""
    seq_len = params.get("seq_len", 96)
    label_len = params.get("label_len", 48)
    pred_len = params.get("pred_len", 1)
    batch_size = params.get("batch_size", 64)

    dataset = InformerDataset(data, seq_len, label_len, pred_len)
    if len(dataset) == 0:
        return np.array([])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            predictions.append(outputs[:, -1, 0].cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([])


def predict_with_lstm(
    model: nn.Module,
    data: np.ndarray,
    params: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Generate predictions with an LSTM model."""
    look_back = params.get("look_back", 20)
    pred_len = params.get("pred_len", 1)
    batch_size = params.get("batch_size", 128)

    dataset = LSTMDataset(data, look_back, pred_len)
    if len(dataset) == 0:
        return np.array([])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output, _ = model(x)
            predictions.append(output.squeeze(-1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([])


def ensemble_predict(
    trained_models: Dict[str, Any],
    normalized_data: Dict[str, Any],
    informer_params: Dict[str, Any],
    lstm_params: Dict[str, Any],
    runtime_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate ensemble predictions using all trained models.

    Args:
        trained_models: Dictionary containing all trained model checkpoints
        normalized_data: Dictionary containing normalized test data
        informer_params: Informer model hyperparameters
        lstm_params: LSTM model hyperparameters
        runtime_params: Runtime settings (device)

    Returns:
        Dictionary containing component predictions and ensemble averages
    """
    logger.info("=" * 60)
    logger.info("INFERENCE - Ensemble Prediction")
    logger.info("=" * 60)

    # Get device
    device_setting = runtime_params.get("device", "auto")
    if device_setting == "auto":
        device = get_device()
    elif device_setting == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Get metadata
    metadata = trained_models["metadata"]
    h_imf_indices = metadata["h_imf_indices"]
    l_imf_indices = metadata["l_imf_indices"]
    n_imfs = metadata["n_imfs"]
    n_models = metadata["n_models"]

    logger.info(f"H-IMF (Informer): {len(h_imf_indices)} components")
    logger.info(f"L-IMF (LSTM): {len(l_imf_indices)} components + RES")
    logger.info(f"Models per component: {n_models}")

    # Get test data (normalized)
    test_imfs = normalized_data["test_imfs_norm"]
    test_residual = normalized_data["test_residual_norm"]

    # Get normalizers for denormalization
    normalizers = normalized_data["normalizers"]

    # Storage for predictions
    component_predictions = {
        "informer": {},  # {imf_idx: {seed: predictions}}
        "lstm": {},
        "residual": {},
    }
    ensemble_predictions = {
        "informer": {},  # {imf_idx: averaged_predictions}
        "lstm": {},
        "residual": None,
    }

    # Predict with Informer models (H-IMF)
    logger.info("-" * 60)
    logger.info("Generating Informer predictions for H-IMF components...")
    for imf_idx in h_imf_indices:
        component_name = f"IMF{imf_idx + 1}"
        test_data = test_imfs[imf_idx]

        component_predictions["informer"][imf_idx] = {}
        all_preds = []

        for seed, checkpoint in trained_models["informer"][imf_idx].items():
            # Create and load model
            model = create_informer_model(informer_params, device)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)

            # Generate predictions
            preds = predict_with_informer(model, test_data, informer_params, device)
            component_predictions["informer"][imf_idx][seed] = preds
            all_preds.append(preds)

            # Clear model from GPU
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Average across seeds
        if all_preds:
            ensemble_predictions["informer"][imf_idx] = np.mean(all_preds, axis=0)
            logger.info(f"  {component_name}: {len(all_preds)} models, {len(ensemble_predictions['informer'][imf_idx])} predictions")

    # Predict with LSTM models (L-IMF)
    logger.info("-" * 60)
    logger.info("Generating LSTM predictions for L-IMF components...")
    for imf_idx in l_imf_indices:
        component_name = f"IMF{imf_idx + 1}"
        test_data = test_imfs[imf_idx]

        component_predictions["lstm"][imf_idx] = {}
        all_preds = []

        for seed, checkpoint in trained_models["lstm"][imf_idx].items():
            # Create and load model
            model = LSTMModel(
                input_size=1,
                hidden_size=lstm_params.get("hidden_size", 4),
                num_layers=lstm_params.get("num_layers", 1),
                output_size=1,
                dropout=lstm_params.get("dropout", 0.1),
                device=device,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)

            # Generate predictions
            preds = predict_with_lstm(model, test_data, lstm_params, device)
            component_predictions["lstm"][imf_idx][seed] = preds
            all_preds.append(preds)

            # Clear model from GPU
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Average across seeds
        if all_preds:
            ensemble_predictions["lstm"][imf_idx] = np.mean(all_preds, axis=0)
            logger.info(f"  {component_name}: {len(all_preds)} models, {len(ensemble_predictions['lstm'][imf_idx])} predictions")

    # Predict residual with LSTM
    logger.info("-" * 60)
    logger.info("Generating LSTM predictions for Residual...")
    all_preds = []
    for seed, checkpoint in trained_models["residual"].items():
        model = LSTMModel(
            input_size=1,
            hidden_size=lstm_params.get("hidden_size", 4),
            num_layers=lstm_params.get("num_layers", 1),
            output_size=1,
            dropout=lstm_params.get("dropout", 0.1),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        preds = predict_with_lstm(model, test_residual, lstm_params, device)
        component_predictions["residual"][seed] = preds
        all_preds.append(preds)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_preds:
        ensemble_predictions["residual"] = np.mean(all_preds, axis=0)
        logger.info(f"  RES: {len(all_preds)} models, {len(ensemble_predictions['residual'])} predictions")

    logger.info("=" * 60)
    logger.info("Inference complete!")
    logger.info("=" * 60)

    return {
        "component_predictions": component_predictions,
        "ensemble_predictions": ensemble_predictions,
        "metadata": metadata,
    }


def reconstruct_predictions(
    predictions: Dict[str, Any],
    normalized_data: Dict[str, Any],
    val_test_decomposition: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct final predictions by denormalizing and summing components.

    Args:
        predictions: Dictionary from ensemble_predict containing ensemble predictions
        normalized_data: Dictionary containing normalizers
        val_test_decomposition: Dictionary containing test residual (actual values)

    Returns:
        Dictionary containing final reconstructed predictions
    """
    logger.info("=" * 60)
    logger.info("RECONSTRUCTION - Denormalize and Sum Components")
    logger.info("=" * 60)

    # Get ensemble predictions and normalizers
    ensemble_preds = predictions["ensemble_predictions"]
    normalizers = normalized_data["normalizers"]
    metadata = predictions["metadata"]

    h_imf_indices = metadata["h_imf_indices"]
    l_imf_indices = metadata["l_imf_indices"]
    n_imfs = metadata["n_imfs"]

    # Find the minimum prediction length (they may differ due to sequence lengths)
    min_len = float("inf")

    for imf_idx in h_imf_indices:
        if imf_idx in ensemble_preds["informer"]:
            min_len = min(min_len, len(ensemble_preds["informer"][imf_idx]))

    for imf_idx in l_imf_indices:
        if imf_idx in ensemble_preds["lstm"]:
            min_len = min(min_len, len(ensemble_preds["lstm"][imf_idx]))

    if ensemble_preds["residual"] is not None:
        min_len = min(min_len, len(ensemble_preds["residual"]))

    if min_len == float("inf"):
        raise ValueError("No predictions found!")

    logger.info(f"Prediction length: {min_len}")

    # Denormalize and sum components
    denorm_predictions = np.zeros(min_len)

    # Process H-IMF (Informer) predictions
    for imf_idx in h_imf_indices:
        if imf_idx in ensemble_preds["informer"]:
            preds = ensemble_preds["informer"][imf_idx][:min_len]
            denorm = normalizers[imf_idx].inverse_transform(preds)
            denorm_predictions += denorm
            logger.info(f"  IMF{imf_idx + 1} (Informer): mean={np.mean(denorm):.4f}")

    # Process L-IMF (LSTM) predictions
    for imf_idx in l_imf_indices:
        if imf_idx in ensemble_preds["lstm"]:
            preds = ensemble_preds["lstm"][imf_idx][:min_len]
            denorm = normalizers[imf_idx].inverse_transform(preds)
            denorm_predictions += denorm
            logger.info(f"  IMF{imf_idx + 1} (LSTM): mean={np.mean(denorm):.4f}")

    # Process residual
    if ensemble_preds["residual"] is not None:
        preds = ensemble_preds["residual"][:min_len]
        denorm = normalizers[-1].inverse_transform(preds)  # Last normalizer is for residual
        denorm_predictions += denorm
        logger.info(f"  RES (LSTM): mean={np.mean(denorm):.4f}")

    # Get actual test values for comparison
    # The actual values need to be aligned with predictions
    # Predictions start after seq_len/look_back, so we need to offset
    test_residual = val_test_decomposition["test_residual"]
    test_imfs = val_test_decomposition["test_imfs"]

    # Reconstruct actual values from IMFs + residual
    actual_values = np.sum(test_imfs, axis=0) + test_residual

    # Align actual values with predictions (offset by max sequence length)
    # Informer uses seq_len=96, LSTM uses look_back=20
    # Use the larger offset for alignment
    offset = 96  # Informer seq_len
    actual_aligned = actual_values[offset:offset + min_len]

    logger.info("-" * 60)
    logger.info(f"Final predictions: {len(denorm_predictions)} samples")
    logger.info(f"Prediction range: {denorm_predictions.min():.2f} to {denorm_predictions.max():.2f}")
    logger.info(f"Actual range: {actual_aligned.min():.2f} to {actual_aligned.max():.2f}")

    # Get test open prices for backtest (aligned with predictions)
    test_open_prices = val_test_decomposition.get("test_open_prices")
    open_prices_aligned = None
    if test_open_prices is not None:
        open_prices_aligned = test_open_prices[offset:offset + min_len]
        logger.info(f"Open prices aligned: {len(open_prices_aligned)} samples")

    # Align test_dates with predictions (same offset as actual values)
    test_dates = normalized_data.get("test_dates")
    test_dates_aligned = None
    if test_dates is not None and len(test_dates) > offset:
        test_dates_aligned = test_dates[offset:offset + min_len]
        logger.info(f"Test dates aligned: {len(test_dates_aligned)} samples")
        if len(test_dates_aligned) > 0:
            logger.info(f"Date range: {test_dates_aligned[0]} to {test_dates_aligned[-1]}")

    return {
        "predictions": denorm_predictions,
        "actual": actual_aligned,
        "open_prices": open_prices_aligned,  # For backtest execution timing
        "test_dates": test_dates_aligned,  # Aligned with predictions
        "n_samples": min_len,
        "offset": offset,
    }

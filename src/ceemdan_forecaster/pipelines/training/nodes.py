# -*- coding: utf-8 -*-
"""Ensemble training nodes.

This module implements training for all 450 models:
- Informer models for H-IMF components (high-frequency)
- LSTM models for L-IMF components (low-frequency)

Each component gets n_models (default: 50) models trained with different seeds.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Add LTSM project to path for model imports
LTSM_ROOT = Path(__file__).parents[5]  # Go up to LTSM directory
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
    """Dataset for Informer model training."""

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
        s_begin = idx + self.seq_len - self.label_len
        s_end = idx + self.seq_len + self.pred_len
        seq_y = self.data[s_begin:s_end]

        # Time features (dummy zeros for compatibility)
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        seq_x = torch.FloatTensor(seq_x).reshape(-1, 1)
        seq_y = torch.FloatTensor(seq_y).reshape(-1, 1)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class LSTMDataset(Dataset):
    """Dataset for LSTM model training."""

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


def train_single_informer(
    train_data: np.ndarray,
    val_data: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single Informer model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    seq_len = params.get("seq_len", 96)
    label_len = params.get("label_len", 48)
    pred_len = params.get("pred_len", 1)
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 10)
    lr = params.get("learning_rate", 0.0001)
    patience = params.get("patience", 3)
    use_amp = params.get("use_amp", True)

    # Create datasets
    train_dataset = InformerDataset(train_data, seq_len, label_len, pred_len)
    val_dataset = InformerDataset(val_data, seq_len, label_len, pred_len)

    if len(train_dataset) == 0:
        raise ValueError(f"Training data too short for seq_len={seq_len}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = create_informer_model(params, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            # CRITICAL: Mask future values in decoder input to prevent look-ahead bias
            # Official Informer implementation zeros out the pred_len portion
            # dec_inp = [label_len known values | pred_len zeros]
            dec_inp = torch.zeros(
                [batch_y.shape[0], pred_len, batch_y.shape[-1]], device=device
            )
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    loss = criterion(outputs, batch_y[:, -pred_len:, :])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y[:, -pred_len:, :])
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x_mark = batch_x_mark.to(device)
                batch_y_mark = batch_y_mark.to(device)

                # CRITICAL: Mask future values in decoder input (same as training)
                dec_inp = torch.zeros(
                    [batch_y.shape[0], pred_len, batch_y.shape[-1]], device=device
                )
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y[:, -pred_len:, :])
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": history,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "seed": seed,
    }


def train_single_lstm(
    train_data: np.ndarray,
    val_data: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single LSTM model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    look_back = params.get("look_back", 20)
    pred_len = params.get("pred_len", 1)
    hidden_size = params.get("hidden_size", 4)
    num_layers = params.get("num_layers", 1)
    dropout = params.get("dropout", 0.1)
    batch_size = params.get("batch_size", 128)
    epochs = params.get("epochs", 100)
    lr = params.get("learning_rate", 0.001)
    patience = params.get("patience", 10)
    use_amp = params.get("use_amp", True)

    # Create datasets
    train_dataset = LSTMDataset(train_data, look_back, pred_len)
    val_dataset = LSTMDataset(val_data, look_back, pred_len)

    if len(train_dataset) == 0:
        raise ValueError(f"Training data too short for look_back={look_back}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = LSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    output, _ = model(x)
                    loss = criterion(output, y.squeeze(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, _ = model(x)
                loss = criterion(output, y.squeeze(-1))
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x)
                loss = criterion(output, y.squeeze(-1))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": history,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "seed": seed,
    }


def train_all_ensembles(
    normalized_data: Dict[str, Any],
    classification: Dict[str, Any],
    informer_params: Dict[str, Any],
    lstm_params: Dict[str, Any],
    ensemble_params: Dict[str, Any],
    runtime_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Train all ensemble models for all IMF components.

    Args:
        normalized_data: Dictionary containing normalized IMF data for all splits
        classification: Dictionary with H-IMF and L-IMF indices
        informer_params: Informer model hyperparameters
        lstm_params: LSTM model hyperparameters
        ensemble_params: Ensemble settings (n_models)
        runtime_params: Runtime settings (device, seed)

    Returns:
        Dictionary containing all trained model checkpoints
    """
    logger.info("=" * 60)
    logger.info("TRAINING - Ensemble Model Training")
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

    # Get classification results
    h_imf_indices = classification["h_imf_indices"]
    l_imf_indices = classification["l_imf_indices"]
    n_imfs = classification["n_imfs"]

    logger.info(f"H-IMF (Informer): {len(h_imf_indices)} components - indices {h_imf_indices}")
    logger.info(f"L-IMF (LSTM): {len(l_imf_indices)} components + RES - indices {l_imf_indices}")

    # Get normalized data
    train_imfs = normalized_data["train_imfs_norm"]
    val_imfs = normalized_data["val_imfs_norm"]
    train_residual = normalized_data["train_residual_norm"]
    val_residual = normalized_data["val_residual_norm"]

    # Ensemble settings
    n_models = ensemble_params.get("n_models", 50)
    base_seed = runtime_params.get("random_seed", 42)

    logger.info(f"Models per component: {n_models}")
    total_models = (len(h_imf_indices) + len(l_imf_indices) + 1) * n_models
    logger.info(f"Total models to train: {total_models}")

    all_checkpoints = {
        "informer": {},  # {imf_index: {seed: checkpoint}}
        "lstm": {},      # {imf_index: {seed: checkpoint}}
        "residual": {},  # {seed: checkpoint}
        "metadata": {
            "n_models": n_models,
            "n_imfs": n_imfs,
            "h_imf_indices": h_imf_indices,
            "l_imf_indices": l_imf_indices,
            "device": str(device),
        },
    }

    models_trained = 0

    # Train Informer models for H-IMF components
    logger.info("-" * 60)
    logger.info("Training Informer models for H-IMF components...")
    for imf_idx in h_imf_indices:
        component_name = f"IMF{imf_idx + 1}"
        logger.info(f"  Training {component_name} ({n_models} models)...")

        train_data = train_imfs[imf_idx]
        val_data = val_imfs[imf_idx]

        all_checkpoints["informer"][imf_idx] = {}

        for seed_offset in range(n_models):
            seed = base_seed + seed_offset
            try:
                checkpoint = train_single_informer(
                    train_data, val_data, informer_params, seed, device
                )
                all_checkpoints["informer"][imf_idx][seed] = checkpoint
                models_trained += 1

                if (seed_offset + 1) % 10 == 0:
                    logger.info(
                        f"    {component_name}: {seed_offset + 1}/{n_models} models, "
                        f"best_val_loss={checkpoint['best_val_loss']:.6f}"
                    )
            except Exception as e:
                logger.error(f"    Failed seed {seed}: {e}")

        # Clear GPU cache after each component
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Train LSTM models for L-IMF components
    logger.info("-" * 60)
    logger.info("Training LSTM models for L-IMF components...")
    for imf_idx in l_imf_indices:
        component_name = f"IMF{imf_idx + 1}"
        logger.info(f"  Training {component_name} ({n_models} models)...")

        train_data = train_imfs[imf_idx]
        val_data = val_imfs[imf_idx]

        all_checkpoints["lstm"][imf_idx] = {}

        for seed_offset in range(n_models):
            seed = base_seed + seed_offset
            try:
                checkpoint = train_single_lstm(
                    train_data, val_data, lstm_params, seed, device
                )
                all_checkpoints["lstm"][imf_idx][seed] = checkpoint
                models_trained += 1

                if (seed_offset + 1) % 10 == 0:
                    logger.info(
                        f"    {component_name}: {seed_offset + 1}/{n_models} models, "
                        f"best_val_loss={checkpoint['best_val_loss']:.6f}"
                    )
            except Exception as e:
                logger.error(f"    Failed seed {seed}: {e}")

        # Clear GPU cache after each component
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Train LSTM models for Residual (always L-IMF)
    logger.info("-" * 60)
    logger.info(f"Training Residual ({n_models} models)...")

    for seed_offset in range(n_models):
        seed = base_seed + seed_offset
        try:
            checkpoint = train_single_lstm(
                train_residual, val_residual, lstm_params, seed, device
            )
            all_checkpoints["residual"][seed] = checkpoint
            models_trained += 1

            if (seed_offset + 1) % 10 == 0:
                logger.info(
                    f"  Residual: {seed_offset + 1}/{n_models} models, "
                    f"best_val_loss={checkpoint['best_val_loss']:.6f}"
                )
        except Exception as e:
            logger.error(f"  Failed seed {seed}: {e}")

    # Clear GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info(f"Training complete! {models_trained}/{total_models} models trained")
    logger.info("=" * 60)

    all_checkpoints["metadata"]["models_trained"] = models_trained
    all_checkpoints["metadata"]["total_expected"] = total_models

    return all_checkpoints

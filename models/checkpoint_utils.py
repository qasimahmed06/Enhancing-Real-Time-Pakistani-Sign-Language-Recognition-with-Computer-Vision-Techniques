"""Helpers for loading checkpoints with common prefix normalisation."""

from __future__ import annotations

from collections.abc import Mapping


def normalize_state_dict(checkpoint):
    """Return a plain state_dict with common wrapper prefixes removed."""
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    normalized = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith("module.") or clean_key.startswith("model."):
            clean_key = clean_key.split(".", 1)[1]
        normalized[clean_key] = value
    return normalized


def infer_num_classes(state_dict, default=36):
    """Infer classifier output size from a checkpoint state_dict."""
    candidate_keys = (
        "fc.weight",
        "classifier.5.weight",
        "classifier.weight",
        "head.weight",
        "last_linear.weight",
    )
    for candidate in candidate_keys:
        if candidate in state_dict:
            return state_dict[candidate].shape[0]

    for key, value in state_dict.items():
        if key.endswith(("fc.weight", "classifier.5.weight", "classifier.weight", "head.weight", "last_linear.weight")):
            return value.shape[0]

    return default


def infer_lstm_input_size(state_dict, default=2048):
    """Infer the LSTM input width from a checkpoint state_dict."""
    if "lstm.weight_ih_l0" in state_dict:
        return state_dict["lstm.weight_ih_l0"].shape[1]

    for key, value in state_dict.items():
        if key.endswith("lstm.weight_ih_l0"):
            return value.shape[1]

    return default
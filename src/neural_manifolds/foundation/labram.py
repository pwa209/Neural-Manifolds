"""Adapter for the pinned official LaBraM implementation.

The adapter intentionally does not vendor or silently update upstream code. The
server bootstrap clones the exact revision in ``configs/models.yaml`` and records
the downloaded checkpoint SHA-256 before this class will load it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_manifolds.foundation.base import (
    EncodedBatch,
    assert_label_firewall,
    validate_frozen_model,
)
from neural_manifolds.preprocessing.eeg import canonicalize_channel_name
from neural_manifolds.provenance import sha256_file

STANDARD_1020 = (
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9H",
    "TTP7H",
    "TPP9H",
    "FTT10H",
    "TPP8H",
    "TPP10H",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)

DEFAULT_REGIONS: Mapping[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "central": ("C3", "Cz", "C4"),
    "temporal": ("T7", "T8"),
    "parietal": ("P7", "P3", "Pz", "P4", "P8"),
    "occipital": ("O1", "O2"),
}


def channel_position_indices(channel_names: Sequence[str]) -> list[int]:
    """Return official LaBraM position indices, including CLS at position zero."""

    normalised = [canonicalize_channel_name(name).upper() for name in channel_names]
    missing = [name for name in normalised if name not in STANDARD_1020]
    if missing:
        raise ValueError(f"channels are absent from LaBraM's standard_1020 map: {missing}")
    if len(set(normalised)) != len(normalised):
        raise ValueError("duplicate channels after normalisation")
    return [0, *(STANDARD_1020.index(name) + 1 for name in normalised)]


def _import_factory(repo: Path, specification: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use module:attribute syntax")
    module_file = repo / (module_name.replace(".", "/") + ".py")
    if not module_file.is_file():
        raise FileNotFoundError(module_file)
    unique_name = f"_neural_manifolds_upstream_{module_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(unique_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import upstream module {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return getattr(module, attribute)


class OfficialLaBraMEncoder:
    """Frozen, final-pre-head feature extractor for official LaBraM-Base."""

    def __init__(
        self,
        *,
        repository: str | Path,
        factory: str,
        checkpoint: str | Path,
        checkpoint_sha256: str,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - foundation environment
            raise RuntimeError("install neural-manifolds[foundation]") from exc
        self.torch = torch
        self.repository = Path(repository).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not checkpoint_sha256 or len(checkpoint_sha256) != 64:
            raise ValueError("an immutable checkpoint SHA-256 is required")
        observed = sha256_file(self.checkpoint)
        if observed.lower() != checkpoint_sha256.lower():
            raise ValueError(
                f"checkpoint checksum mismatch: expected {checkpoint_sha256}, got {observed}"
            )
        factory_function = _import_factory(self.repository, factory)
        model = factory_function(num_classes=0)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain a state dictionary")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint/model mismatch; missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        model.requires_grad_(False)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        validate_frozen_model(self.model)
        self.checkpoint_hash = observed

    def encode(
        self,
        windows_volts: np.ndarray,
        channel_names: Sequence[str],
        *,
        metadata_fields: Sequence[str] = (),
        batch_size: int = 64,
        regions: Mapping[str, Sequence[str]] = DEFAULT_REGIONS,
    ) -> EncodedBatch:
        """Encode windows shaped ``windows x channels x samples`` at 200 Hz."""

        assert_label_firewall(metadata_fields)
        x = np.asarray(windows_volts, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] != len(channel_names):
            raise ValueError("windows must be windows x channels x samples")
        if x.shape[2] % 200:
            raise ValueError("LaBraM requires an integer number of 200-sample patches")
        if not np.all(np.isfinite(x)):
            raise ValueError("windows contain non-finite samples")
        normalised_names = tuple(canonicalize_channel_name(name) for name in channel_names)
        input_chans = channel_position_indices(normalised_names)
        patches = x.reshape(x.shape[0], x.shape[1], x.shape[2] // 200, 200) * 1e6
        outputs: list[np.ndarray] = []
        torch = self.torch
        validate_frozen_model(self.model)
        with torch.inference_mode():
            for start in range(0, patches.shape[0], batch_size):
                batch = torch.from_numpy(patches[start : start + batch_size]).to(self.device)
                tokens = self.model.forward_features(
                    batch,
                    input_chans=input_chans,
                    return_patch_tokens=True,
                )
                outputs.append(tokens.detach().float().cpu().numpy())
        flattened = np.concatenate(outputs, axis=0)
        token_grid = flattened.reshape(
            patches.shape[0], patches.shape[1], patches.shape[2], flattened.shape[-1]
        )
        global_states = token_grid.mean(axis=(1, 2))
        regional_states: dict[str, np.ndarray] = {}
        for region, names in regions.items():
            indices = [index for index, name in enumerate(normalised_names) if name in names]
            if indices:
                regional_states[region] = token_grid[:, indices].mean(axis=(1, 2))
        return EncodedBatch(
            global_states=global_states,
            regional_states=regional_states,
            patch_tokens=token_grid,
            channel_names=normalised_names,
            metadata={
                "encoder": "labram_base",
                "checkpoint_sha256": self.checkpoint_hash,
                "units_in": "volts",
                "units_model": "microvolts",
                "patch_samples": 200,
                "pooling": "mean_valid_channel_patch_tokens",
            },
        )

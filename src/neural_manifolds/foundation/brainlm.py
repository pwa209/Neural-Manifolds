"""Audited, frozen adapter for the pinned official BrainLM ViT-MAE release.

The released ``vitmae_111M`` checkpoint is not the older coordinate-token
``BrainLMForPretraining`` class.  Its checked configuration declares a padded
ViT-MAE image model.  This adapter follows the zero-shot path in the pinned
upstream tutorial: 200 time points by 424 A424 parcels, parcels stably ordered
by the tutorial's ``Y`` coordinate, three repeated channels, division by the
published scale constant, and symmetric ``-1`` padding to 432 by 432.

No network access occurs here.  Source, configuration, and safetensors weights
must already have been materialised and verified by the server bootstrap.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_manifolds.foundation.base import assert_label_firewall, validate_frozen_model
from neural_manifolds.provenance import sha256_file

OFFICIAL_REPOSITORY = "https://github.com/vandijklab/BrainLM.git"
USAGE_LICENSE = "CC-BY-NC-ND-4.0"
PARCELLATION = "UKB_424"
N_PARCELS = 424
MODEL_WINDOW_TIMEPOINTS = 200
WINDOW_STEP_TIMEPOINTS = 20
TUTORIAL_SCALE = 5.6430855
SOURCE_MODEL_FILE = "brainlm_mae/modeling_vit_mae_with_padding.py"


@dataclass(frozen=True)
class BrainLMEncoding:
    """Window-wise label-free BrainLM coordinates for one fMRI segment."""

    global_states: np.ndarray
    window_starts: np.ndarray
    window_stops: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        states = np.asarray(self.global_states)
        starts = np.asarray(self.window_starts)
        stops = np.asarray(self.window_stops)
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError("global_states must be at least two windows by features")
        if starts.ndim != 1 or stops.ndim != 1 or starts.size != states.shape[0]:
            raise ValueError("window indices must match the encoded window count")
        if stops.size != starts.size or np.any(starts < 0) or np.any(stops <= starts):
            raise ValueError("window indices are invalid")
        if np.any(starts[1:] <= starts[:-1]):
            raise ValueError("window starts must increase strictly")
        if not np.all(np.isfinite(states)):
            raise ValueError("global_states contain non-finite values")


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed for BrainLM source: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_official_source(
    repository: str | Path,
    *,
    expected_revision: str,
    expected_repository: str = OFFICIAL_REPOSITORY,
) -> Path:
    """Require the exact clean upstream checkout selected in ``models.yaml``."""

    source = Path(repository).resolve(strict=True)
    if not (source / ".git").is_dir():
        raise ValueError(f"BrainLM source is not a Git checkout: {source}")
    if len(expected_revision) != 40 or any(
        char not in "0123456789abcdef" for char in expected_revision
    ):
        raise ValueError("BrainLM revision must be a full lowercase Git object id")
    revision = _run_git(source, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(f"BrainLM source revision is {revision}, expected {expected_revision}")
    origin = _run_git(source, "remote", "get-url", "origin")
    if origin != expected_repository:
        raise ValueError(f"BrainLM source origin is {origin!r}, expected {expected_repository!r}")
    changed = _run_git(source, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise ValueError("BrainLM checkout contains modified tracked source files")
    model_file = source / SOURCE_MODEL_FILE
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    return source


def load_ukb424_coordinates(path: str | Path) -> np.ndarray:
    """Load an ordered A424 coordinate table and enforce its 424-by-3 contract."""

    source = Path(path).resolve(strict=True)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"coordinates"}:
                raise ValueError("coordinate NPZ must contain only the 'coordinates' array")
            values = archive["coordinates"]
    else:
        delimiter = "," if suffix == ".csv" else None
        values = np.genfromtxt(source, delimiter=delimiter, dtype=np.float64)
        if values.ndim == 2 and np.isnan(values[0]).any():
            values = values[1:]
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (N_PARCELS, 4):
        index = array[:, 0]
        zero_based = np.arange(N_PARCELS, dtype=np.float64)
        one_based = zero_based + 1.0
        if not (np.array_equal(index, zero_based) or np.array_equal(index, one_based)):
            raise ValueError("A424 coordinate index must be exactly 0..423 or 1..424")
        array = array[:, 1:]
    if array.shape != (N_PARCELS, 3):
        raise ValueError(f"UKB_424 coordinates must be 424 x 3, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("UKB_424 coordinates contain non-finite values")
    if np.unique(array, axis=0).shape[0] != N_PARCELS:
        raise ValueError("UKB_424 coordinate rows must be unique")
    return array


def validate_parcel_timeseries(values: np.ndarray) -> np.ndarray:
    """Require the unambiguous time-by-UKB_424 orientation."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != N_PARCELS:
        raise ValueError(f"parcel time series must be time x {N_PARCELS}, got {array.shape}")
    if array.shape[0] < MODEL_WINDOW_TIMEPOINTS + WINDOW_STEP_TIMEPOINTS:
        raise ValueError(
            f"BrainLM trajectory requires at least {MODEL_WINDOW_TIMEPOINTS + WINDOW_STEP_TIMEPOINTS} "
            "time points (two distinct 200-TR windows)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("parcel time series contains non-finite values")
    return array


def window_starts(
    n_timepoints: int,
    *,
    length: int = MODEL_WINDOW_TIMEPOINTS,
    step: int = WINDOW_STEP_TIMEPOINTS,
) -> np.ndarray:
    """Return deterministic sliding-window starts, including the last valid window."""

    if length <= 0 or step <= 0 or n_timepoints < length:
        raise ValueError("invalid BrainLM window length, step, or recording duration")
    starts = list(range(0, n_timepoints - length + 1, step))
    final = n_timepoints - length
    if starts[-1] != final:
        starts.append(final)
    return np.asarray(starts, dtype=np.int64)


def _load_source_module(repository: Path) -> Any:
    model_file = repository / SOURCE_MODEL_FILE
    unique = f"_neural_manifolds_brainlm_{sha256_file(model_file)[:16]}"
    if unique in sys.modules:
        return sys.modules[unique]
    specification = importlib.util.spec_from_file_location(unique, model_file)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import pinned BrainLM module: {model_file}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[unique] = module
    specification.loader.exec_module(module)
    return module


def _validate_checkpoint_config(document: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "vit_mae",
        "architectures": ["ViTMAEForPreTraining"],
        "image_size": [432, 432],
        "patch_size": 16,
        "num_channels": 3,
        "hidden_size": 768,
        "num_hidden_layers": 12,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"pinned BrainLM checkpoint has unexpected {field}: "
                f"{document.get(field)!r} (expected {value!r})"
            )


class OfficialBrainLMEncoder:
    """Frozen final-four-layer CLS extractor for ``vitmae_111M``."""

    def __init__(
        self,
        *,
        repository: str | Path,
        source_revision: str,
        checkpoint_config: str | Path,
        checkpoint_config_sha256: str,
        checkpoint_weights: str | Path,
        checkpoint_weights_sha256: str,
        checkpoint_weights_size: int | None = None,
        repository_url: str = OFFICIAL_REPOSITORY,
        parcellation: str = PARCELLATION,
        usage_license: str = USAGE_LICENSE,
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        if parcellation != PARCELLATION:
            raise ValueError(f"BrainLM requires parcellation {PARCELLATION}")
        if usage_license != USAGE_LICENSE:
            raise ValueError(f"BrainLM weights require {USAGE_LICENSE}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.repository = verify_official_source(
            repository,
            expected_revision=source_revision,
            expected_repository=repository_url,
        )
        self.source_revision = source_revision
        self.config_path = Path(checkpoint_config).resolve(strict=True)
        self.weights_path = Path(checkpoint_weights).resolve(strict=True)
        if self.config_path.name != "config.json":
            raise ValueError("BrainLM configuration must be the pinned config.json")
        if self.weights_path.name != "model.safetensors":
            raise ValueError("BrainLM weights must use the pinned safe model.safetensors")
        for label, expected_hash, path in (
            ("configuration", checkpoint_config_sha256, self.config_path),
            ("weights", checkpoint_weights_sha256, self.weights_path),
        ):
            if len(expected_hash) != 64:
                raise ValueError(f"BrainLM {label} requires an immutable SHA-256")
            observed = sha256_file(path)
            if observed != expected_hash.lower():
                raise ValueError(
                    f"BrainLM {label} checksum mismatch: expected {expected_hash}, got {observed}"
                )
        if checkpoint_weights_size is not None and self.weights_path.stat().st_size != int(
            checkpoint_weights_size
        ):
            raise ValueError("BrainLM safetensors size differs from the pinned model specification")
        document = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("BrainLM configuration root must be a JSON object")
        _validate_checkpoint_config(document)

        try:
            import torch
            from safetensors.torch import load_file
            from transformers import ViTMAEConfig
        except ImportError as exc:  # pragma: no cover - server fMRI environment
            raise RuntimeError("install neural-manifolds[foundation,fmri]") from exc
        self.torch = torch
        self.device = torch.device(device)
        config = ViTMAEConfig.from_json_file(str(self.config_path))
        module = _load_source_module(self.repository)
        model = module.ViTMAEForPreTraining(config)
        state = load_file(str(self.weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
        # The checkpoint was trained with masking, but a coordinate transform must
        # see every token. Identity noise preserves the official raster order.
        model.config.mask_ratio = 0.0
        model.vit.config.mask_ratio = 0.0
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)
        validate_frozen_model(self.model)
        self.batch_size = int(batch_size)
        self.config_sha256 = checkpoint_config_sha256.lower()
        self.weights_sha256 = checkpoint_weights_sha256.lower()
        self.metadata: dict[str, Any] = {
            "encoder": "brainlm_vitmae_111M",
            "source_repository": repository_url,
            "source_revision": source_revision,
            "checkpoint_config_sha256": self.config_sha256,
            "checkpoint_weights_sha256": self.weights_sha256,
            "checkpoint_format": "safetensors",
            "weights_frozen": True,
            "label_free": True,
            "parcellation": PARCELLATION,
            "usage_license": USAGE_LICENSE,
            "commercial_use": False,
            "derivative_redistribution": False,
            "layer_pooling": "mean_final_four_encoder_layers_then_layernorm_cls",
            "inference_mask_ratio": 0.0,
            "window_timepoints": MODEL_WINDOW_TIMEPOINTS,
            "window_step_timepoints": WINDOW_STEP_TIMEPOINTS,
            "parcel_order": "stable_argsort_A424_Y_coordinate_as_pinned_tutorial",
            "tutorial_scale": TUTORIAL_SCALE,
            "padding": "symmetric_constant_minus_one_to_3x432x432",
        }

    @classmethod
    def from_environment(
        cls,
        model_spec: Mapping[str, Any],
        *,
        device: str = "cuda",
        batch_size: int = 4,
        environment: Mapping[str, str] | None = None,
    ) -> OfficialBrainLMEncoder:
        """Construct only from bootstrap paths and the strict ``models.yaml`` entry."""

        env = os.environ if environment is None else environment
        source_value = env.get("NEURAL_MANIFOLDS_BRAINLM_SOURCE")
        checkpoint_value = env.get("NEURAL_MANIFOLDS_BRAINLM_CHECKPOINT_DIR")
        if not source_value or not checkpoint_value:
            raise RuntimeError("verified BrainLM source/checkpoint environment paths are missing")
        if env.get("NEURAL_MANIFOLDS_BRAINLM_LICENSE") != USAGE_LICENSE:
            raise RuntimeError("BrainLM licence environment receipt is missing or unexpected")
        files = model_spec.get("checkpoint_files")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            raise ValueError("BrainLM model specification has no checkpoint_files list")
        by_name = {str(item.get("name")): item for item in files if isinstance(item, Mapping)}
        variant = model_spec.get("checkpoint_variant")
        config_name = f"{variant}/config.json"
        weights_name = f"{variant}/model.safetensors"
        if set(by_name) != {config_name, weights_name}:
            raise ValueError("BrainLM model specification must contain only config and safetensors")
        root = Path(checkpoint_value).resolve(strict=True)
        config_item = by_name[config_name]
        weights_item = by_name[weights_name]
        return cls(
            repository=source_value,
            source_revision=str(model_spec.get("revision")),
            checkpoint_config=root / config_name,
            checkpoint_config_sha256=str(config_item.get("sha256")),
            checkpoint_weights=root / weights_name,
            checkpoint_weights_sha256=str(weights_item.get("sha256")),
            checkpoint_weights_size=int(weights_item["size"]),
            repository_url=str(model_spec.get("repository")),
            parcellation=str(model_spec.get("parcellation")),
            usage_license=str(model_spec.get("usage_license")),
            device=device,
            batch_size=batch_size,
        )

    def encode(
        self,
        parcel_timeseries: np.ndarray,
        coordinates: np.ndarray,
        *,
        metadata_fields: Sequence[str] = (),
    ) -> BrainLMEncoding:
        """Encode globally normalised time-by-424 data into sliding CLS states."""

        assert_label_firewall(metadata_fields)
        values = validate_parcel_timeseries(parcel_timeseries)
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (N_PARCELS, 3) or not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must satisfy the ordered UKB_424 424 x 3 contract")
        starts = window_starts(values.shape[0])
        reorder = np.argsort(coords[:, 1], kind="stable")
        outputs: list[np.ndarray] = []
        torch = self.torch
        patch_grid = 432 // 16
        validate_frozen_model(self.model)
        with torch.inference_mode():
            for offset in range(0, starts.size, self.batch_size):
                current = starts[offset : offset + self.batch_size]
                windows = np.stack(
                    [values[start : start + MODEL_WINDOW_TIMEPOINTS].T for start in current]
                )
                windows = windows[:, reorder] / np.float32(TUTORIAL_SCALE)
                images = np.repeat(windows[:, None], 3, axis=1)
                images = np.pad(
                    images,
                    ((0, 0), (0, 0), (4, 4), (116, 116)),
                    mode="constant",
                    constant_values=-1.0,
                )
                pixel_values = torch.from_numpy(images).to(self.device)
                noise = (
                    torch.arange(patch_grid * patch_grid, device=self.device)
                    .unsqueeze(0)
                    .repeat(pixel_values.shape[0], 1)
                    .to(pixel_values.dtype)
                )
                encoded = self.model.vit(
                    pixel_values=pixel_values,
                    noise=noise,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = encoded.hidden_states
                if hidden is None or len(hidden) < 4:
                    raise RuntimeError("BrainLM did not expose the configured final four layers")
                pooled = torch.stack(hidden[-4:], dim=0).mean(dim=0)
                pooled = self.model.vit.layernorm(pooled)
                outputs.append(pooled[:, 0].detach().float().cpu().numpy())
        states = np.concatenate(outputs, axis=0)
        return BrainLMEncoding(
            global_states=states,
            window_starts=starts,
            window_stops=starts + MODEL_WINDOW_TIMEPOINTS,
            metadata={**self.metadata, "encoded_windows": int(starts.size)},
        )

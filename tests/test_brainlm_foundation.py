from pathlib import Path

import numpy as np
import pytest

from neural_manifolds.foundation.brainlm import (
    N_PARCELS,
    BrainLMEncoding,
    OfficialBrainLMEncoder,
    load_ukb424_coordinates,
    validate_parcel_timeseries,
    window_starts,
)


def test_brainlm_encoding_requires_aligned_windows() -> None:
    with pytest.raises(ValueError, match="match"):
        BrainLMEncoding(
            global_states=np.ones((3, 4)),
            window_starts=np.asarray([0, 20]),
            window_stops=np.asarray([200, 220]),
            metadata={},
        )


def test_ukb424_coordinates_accept_ordered_index_column(tmp_path: Path) -> None:
    coordinates = np.column_stack(
        (
            np.arange(1, N_PARCELS + 1),
            np.linspace(-40, 40, N_PARCELS),
            np.linspace(-60, 60, N_PARCELS),
            np.linspace(-30, 70, N_PARCELS),
        )
    )
    path = tmp_path / "A424_Coordinates.dat"
    np.savetxt(path, coordinates)
    observed = load_ukb424_coordinates(path)
    assert observed.shape == (N_PARCELS, 3)
    np.testing.assert_allclose(observed, coordinates[:, 1:])


def test_brainlm_timeseries_orientation_is_not_guessed() -> None:
    with pytest.raises(ValueError, match="time x 424"):
        validate_parcel_timeseries(np.ones((N_PARCELS, 240)))


def test_windows_include_final_non_grid_aligned_window() -> None:
    starts = window_starts(245, length=200, step=20)
    np.testing.assert_array_equal(starts, np.asarray([0, 20, 40, 45]))


def test_environment_constructor_requires_bootstrap_receipts() -> None:
    with pytest.raises(RuntimeError, match="environment paths"):
        OfficialBrainLMEncoder.from_environment(
            {"checkpoint_files": []},
            environment={},
            device="cpu",
        )

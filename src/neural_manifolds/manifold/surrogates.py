"""Null families that preserve selected marginals while disrupting organisation."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import (
    FloatArray,
    as_float_matrix,
    check_random_generator,
    encode_states,
    validate_segment_ids,
)


def _time_series_matrix(x: ArrayLike, *, name: str = "x") -> tuple[FloatArray, bool]:
    array = np.asarray(x)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    matrix = as_float_matrix(array, name=name, min_samples=4)
    return matrix, was_vector


def phase_randomized_surrogate(
    x: ArrayLike,
    *,
    random_state: int | np.random.Generator | None = 0,
    shared_phases: bool = False,
) -> NDArray[np.float64]:
    """Randomise Fourier phase while preserving each feature's power spectrum.

    Independent phases (default) disrupt cross-feature trajectory organisation.
    ``shared_phases=True`` instead preserves relative cross-spectral phases and is
    useful as a stricter univariate temporal null.
    """

    matrix, was_vector = _time_series_matrix(x)
    generator = check_random_generator(random_state)
    spectrum = np.fft.rfft(matrix, axis=0)
    n_frequencies = spectrum.shape[0]
    first_random = 1  # DC must remain real and unchanged.
    last_random = n_frequencies - 1 if matrix.shape[0] % 2 == 0 else n_frequencies
    n_random = max(last_random - first_random, 0)
    if n_random:
        phase_shape = (n_random, 1 if shared_phases else matrix.shape[1])
        phase = generator.uniform(0.0, 2.0 * np.pi, size=phase_shape)
        spectrum[first_random:last_random] *= np.exp(1j * phase)
    surrogate = np.fft.irfft(spectrum, n=matrix.shape[0], axis=0)
    return surrogate[:, 0] if was_vector else surrogate


def iaaft_surrogate(
    x: ArrayLike,
    *,
    n_iterations: int = 200,
    tolerance: float = 1e-8,
    random_state: int | np.random.Generator | None = 0,
) -> NDArray[np.float64]:
    """Iterative amplitude-adjusted Fourier surrogate for each feature.

    The final rank projection preserves the empirical marginal distribution
    exactly; iterative spectral projection preserves the periodogram
    approximately. Features are randomised independently to disrupt multivariate
    geometry.
    """

    matrix, was_vector = _time_series_matrix(x)
    if not isinstance(n_iterations, Integral) or isinstance(n_iterations, bool) or n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer")
    if not isinstance(tolerance, Real) or tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    generator = check_random_generator(random_state)
    output = np.empty_like(matrix)
    for feature in range(matrix.shape[1]):
        original = matrix[:, feature]
        target_sorted = np.sort(original)
        target_amplitude = np.abs(np.fft.rfft(original))
        candidate = generator.permutation(original)
        previous_error = np.inf
        for _ in range(int(n_iterations)):
            candidate_spectrum = np.fft.rfft(candidate)
            magnitude = np.abs(candidate_spectrum)
            phases = np.divide(
                candidate_spectrum,
                magnitude,
                out=np.ones_like(candidate_spectrum),
                where=magnitude > 0.0,
            )
            spectral_projection = np.fft.irfft(target_amplitude * phases, n=original.size)
            order = np.argsort(spectral_projection, kind="mergesort")
            candidate = np.empty_like(original)
            candidate[order] = target_sorted
            current_amplitude = np.abs(np.fft.rfft(candidate))
            denominator = max(float(np.linalg.norm(target_amplitude)), 1e-15)
            error = float(np.linalg.norm(current_amplitude - target_amplitude) / denominator)
            if abs(previous_error - error) <= tolerance:
                break
            previous_error = error
        output[:, feature] = candidate
    return output[:, 0] if was_vector else output


def block_permutation(
    x: ArrayLike,
    *,
    block_size: int,
    segment_ids: ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> NDArray[np.float64]:
    """Permute contiguous blocks within segments while preserving block interiors."""

    matrix, was_vector = _time_series_matrix(x)
    if not isinstance(block_size, Integral) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    segments = validate_segment_ids(segment_ids, matrix.shape[0])
    generator = check_random_generator(random_state)
    output = np.empty_like(matrix)
    starts = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1]
    stops = np.r_[starts[1:], matrix.shape[0]]
    for start, stop in zip(starts, stops, strict=True):
        blocks = [
            np.arange(block_start, min(block_start + int(block_size), stop))
            for block_start in range(int(start), int(stop), int(block_size))
        ]
        ordering = generator.permutation(len(blocks))
        source_indices = np.concatenate([blocks[index] for index in ordering])
        output[start:stop] = matrix[source_indices]
    return output[:, 0] if was_vector else output


def permute_channels(
    x: ArrayLike,
    *,
    random_state: int | np.random.Generator | None = 0,
    return_permutation: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Permute feature identities while preserving every univariate time series."""

    matrix = as_float_matrix(x, name="x", min_samples=2)
    generator = check_random_generator(random_state)
    permutation = np.asarray(generator.permutation(matrix.shape[1]), dtype=np.int64)
    result = matrix[:, permutation]
    if return_permutation:
        return result, permutation
    return result


def rotate_channels(
    x: ArrayLike,
    *,
    random_state: int | np.random.Generator | None = 0,
    return_rotation: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a deterministic-seeded random orthogonal feature-space rotation."""

    matrix = as_float_matrix(x, name="x", min_samples=2)
    generator = check_random_generator(random_state)
    gaussian = generator.normal(size=(matrix.shape[1], matrix.shape[1]))
    rotation, triangular = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    rotation *= signs
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    result = matrix @ rotation
    if return_rotation:
        return result, rotation
    return result


def covariance_matched_surrogate(
    x: ArrayLike,
    *,
    random_state: int | np.random.Generator | None = 0,
) -> FloatArray:
    """Generate Gaussian samples with exactly the input sample mean/covariance."""

    matrix = as_float_matrix(x, name="x", min_samples=3)
    n_samples, n_features = matrix.shape
    mean = np.mean(matrix, axis=0)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    if n_features == 1:
        covariance = np.asarray([[float(covariance)]])
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    tolerance = 1e-12 * max(float(eigenvalues[-1]), 1.0)
    keep = eigenvalues > tolerance
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return np.broadcast_to(mean, matrix.shape).copy()
    if rank > n_samples - 1:
        raise ValueError("sample covariance rank exceeds the centred sample subspace")
    generator = check_random_generator(random_state)
    gaussian = generator.normal(size=(n_samples, rank))
    gaussian -= np.mean(gaussian, axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(gaussian, mode="reduced")
    # QR of centred columns remains orthogonal to the all-ones vector to rounding.
    positive_values = eigenvalues[keep]
    positive_vectors = eigenvectors[:, keep]
    centred = (
        orthonormal @ np.diag(np.sqrt((n_samples - 1.0) * positive_values)) @ positive_vectors.T
    )
    centred -= np.mean(centred, axis=0, keepdims=True)
    return centred + mean


def dwell_matched_state_surrogate(
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
    max_attempts: int = 10_000,
) -> NDArray:
    """Randomise run order while preserving each labelled dwell run exactly."""

    labels, encoded = encode_states(states)
    segments = validate_segment_ids(segment_ids, encoded.size)
    if not isinstance(max_attempts, Integral) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    generator = check_random_generator(random_state)
    output = np.empty_like(encoded)
    starts = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1]
    stops = np.r_[starts[1:], encoded.size]
    for segment_start, segment_stop in zip(starts, stops, strict=True):
        segment = encoded[segment_start:segment_stop]
        run_starts = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1]
        run_stops = np.r_[run_starts[1:], segment.size]
        run_states = segment[run_starts]
        run_lengths = run_stops - run_starts
        if run_states.size <= 1:
            output[segment_start:segment_stop] = segment
            continue
        accepted: NDArray[np.int64] | None = None
        for _ in range(int(max_attempts)):
            proposal = np.asarray(generator.permutation(run_states.size), dtype=np.int64)
            proposed_states = run_states[proposal]
            if np.all(proposed_states[1:] != proposed_states[:-1]):
                accepted = proposal
                break
        if accepted is None:
            raise RuntimeError("could not construct a non-merging dwell-matched surrogate")
        reconstructed = np.concatenate(
            [np.full(run_lengths[index], run_states[index], dtype=np.int64) for index in accepted]
        )
        output[segment_start:segment_stop] = reconstructed
    return labels[output]


def state_space_surrogate(
    x: ArrayLike,
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[FloatArray, NDArray]:
    """Match global covariance and dwell runs while removing state geometry.

    A single generator feeds both components, making the joint null exactly
    reproducible without accidentally reusing identical random streams.
    """

    matrix = as_float_matrix(x, name="x", min_samples=3)
    encode_states(states, n_samples=matrix.shape[0])
    generator = check_random_generator(random_state)
    geometry_null = covariance_matched_surrogate(matrix, random_state=generator)
    dwell_null = dwell_matched_state_surrogate(
        states, segment_ids=segment_ids, random_state=generator
    )
    return geometry_null, dwell_null

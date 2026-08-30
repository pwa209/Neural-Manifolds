"""Figure-specific visual arguments built only from validated source tables."""

from __future__ import annotations

import hashlib
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neural_manifolds.manifold.clinical_reference import WAKE_REGIME_LLR

from .config import FigureConfig, FigureContract
from .style import add_panel_label, apply_publication_style, figure_size


@dataclass(frozen=True)
class RenderedFigure:
    figure: Any
    source_panels: dict[str, pd.DataFrame]


def _pyplot() -> Any:
    # Importing here guarantees that config and source validation can fail before a
    # graphics device is opened. style.py has already selected the headless Agg backend.
    import matplotlib.pyplot as plt

    return plt


def _seed_for(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def _biological_key(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset_id"].astype(str) + "::" + frame["participant_id"].astype(str)


def _participant_values(
    frame: pd.DataFrame,
    *,
    group_columns: Iterable[str],
    value_column: str,
) -> pd.DataFrame:
    columns = [*group_columns, "dataset_id", "participant_id"]
    grouped = frame.groupby(columns, sort=True, dropna=False)
    aggregated = grouped[value_column].mean().reset_index()
    for provenance_column in ("source_artifact_sha256", "source_table_sha256"):
        if provenance_column in frame.columns:
            provenance = (
                grouped[provenance_column]
                .agg(lambda values: ";".join(sorted(set(values.astype(str)))))
                .reset_index(name=provenance_column)
            )
            aggregated = aggregated.merge(provenance, on=columns, validate="one_to_one")
    aggregated = aggregated.sort_values(columns, kind="mergesort").reset_index(drop=True)
    aggregated["biological_participant"] = _biological_key(aggregated)
    return aggregated


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    label: str,
    repetitions: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(_seed_for(label))
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def _summary(
    participant_values: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, part in participant_values.groupby(grouping, sort=True, dropna=False):
        key_tuple = (keys,) if len(group_columns) == 1 else tuple(keys)
        key_values = dict(zip(group_columns, key_tuple, strict=True))
        values = part[value_column].to_numpy(dtype=float)
        label = "|".join(f"{key}={value}" for key, value in key_values.items())
        mean, low, high = _bootstrap_mean_interval(values, label=label, repetitions=repetitions)
        rows.append(
            {
                **key_values,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_participants": int(part["biological_participant"].nunique()),
                "interval": "participant bootstrap 95% percentile",
            }
        )
    return pd.DataFrame(rows)


def _short(value: object, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(value).replace("_", " "), width=width))


def _category_colors(categories: list[str], palette: dict[str, str]) -> dict[str, str]:
    colors = [
        palette["blue"],
        palette["orange"],
        palette["teal"],
        palette["violet"],
        palette["red"],
        palette["neutral"],
        palette["blue_light"],
    ]
    return {category: colors[index % len(colors)] for index, category in enumerate(categories)}


def _display_values(
    values: Iterable[object],
    *,
    priority: Iterable[object] = (),
    maximum: int | None = None,
) -> list[str]:
    """Select configured main-panel values while retaining full source tables."""

    observed = {str(value) for value in values}
    ordered = [str(value) for value in priority if str(value) in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    if maximum is None:
        return ordered
    if maximum < 1:
        raise ValueError("configured display maximum must be positive")
    return ordered[:maximum]


def _ordered_summary(frame: pd.DataFrame, *, column: str, order: Sequence[str]) -> pd.DataFrame:
    positions = {value: index for index, value in enumerate(order)}
    result = frame.copy()
    result["__display_order"] = result[column].astype(str).map(positions)
    return result.sort_values("__display_order", kind="mergesort").drop(columns="__display_order")


def _dot_interval(
    ax: Any,
    participant_values: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    palette: dict[str, str],
    ylabel: str | None = None,
    horizontal: bool = False,
) -> None:
    categories = [str(value) for value in summary[category_column]]
    colors = _category_colors(categories, palette)
    for position, category in enumerate(categories):
        points = participant_values.loc[
            participant_values[category_column].astype(str) == category, value_column
        ].to_numpy(dtype=float)
        rng = np.random.default_rng(_seed_for(f"jitter:{category_column}:{category}"))
        jitter = rng.uniform(-0.15, 0.15, size=len(points))
        stat = summary.loc[summary[category_column].astype(str) == category].iloc[0]
        color = colors[category]
        if horizontal:
            ax.scatter(points, position + jitter, s=12, color=color, alpha=0.55, linewidth=0)
            ax.plot(
                [stat["ci_low"], stat["ci_high"]],
                [position, position],
                color=palette["dark"],
                linewidth=1.3,
                zorder=4,
            )
            ax.scatter(stat["mean"], position, s=25, color=palette["dark"], marker="D", zorder=5)
            ax.text(
                0.99,
                position,
                f"n={int(stat['n_participants'])}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6,
            )
        else:
            ax.scatter(position + jitter, points, s=12, color=color, alpha=0.55, linewidth=0)
            ax.plot(
                [position, position],
                [stat["ci_low"], stat["ci_high"]],
                color=palette["dark"],
                linewidth=1.3,
                zorder=4,
            )
            ax.scatter(position, stat["mean"], s=25, color=palette["dark"], marker="D", zorder=5)
            ax.text(
                position,
                0.99,
                f"n={int(stat['n_participants'])}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.7,
            )
    if horizontal:
        ax.set_yticks(range(len(categories)), [_short(value) for value in categories])
        if ylabel:
            ax.set_xlabel(ylabel)
    else:
        ax.set_xticks(range(len(categories)), [_short(value) for value in categories])
        if ylabel:
            ax.set_ylabel(ylabel)
    ax.axhline(0, color=palette["pale"], linewidth=0.7, zorder=0) if not horizontal else ax.axvline(
        0, color=palette["pale"], linewidth=0.7, zorder=0
    )


def _heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    palette: dict[str, str],
    colorbar_label: str,
    symmetric: bool = True,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-12)
    kwargs: dict[str, object] = {}
    if symmetric:
        kwargs.update(vmin=-limit, vmax=limit, cmap="RdBu_r")
    else:
        kwargs.update(cmap="viridis")
    image = ax.imshow(matrix, aspect="auto", interpolation="none", **kwargs)
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), [_short(value, 22) for value in row_labels])
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(colorbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure_1(
    profiles: pd.DataFrame, config: FigureConfig, contract: FigureContract
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.8, 1.15])
    ax_workflow = fig.add_subplot(grid[0, 0])
    ax_axes = fig.add_subplot(grid[0, 1])
    ax_coverage = fig.add_subplot(grid[0, 2])

    workflow = [
        ("recordings", "EEG / TMS-EEG / fMRI"),
        ("frozen coordinates", "LaBraM; BrainLM secondary"),
        ("five-axis profile", "R · M · D · A · P"),
        ("transfer", "perturbation · held-out clinical"),
    ]
    y_positions = np.linspace(0.87, 0.13, len(workflow))
    for index, ((title, subtitle), y_value) in enumerate(zip(workflow, y_positions, strict=True)):
        ax_workflow.text(
            0.5,
            y_value,
            f"{title}\n{subtitle}",
            ha="center",
            va="center",
            transform=ax_workflow.transAxes,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": config.palette["pale"] if index != 2 else config.palette["blue_light"],
                "edgecolor": config.palette["dark"],
                "linewidth": 0.7,
            },
        )
        if index < len(workflow) - 1:
            ax_workflow.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.07),
                xytext=(0.5, y_value - 0.07),
                xycoords=ax_workflow.transAxes,
                arrowprops={"arrowstyle": "->", "color": config.palette["neutral"], "lw": 0.9},
            )
    ax_workflow.text(
        0.02,
        0.01,
        "representation weights frozen · label-free fitting",
        transform=ax_workflow.transAxes,
        fontsize=5.8,
        color=config.palette["neutral"],
    )
    ax_workflow.axis("off")
    add_panel_label(ax_workflow, "a", font_size=float(config.export["panel_label_font_pt"]))

    for index, axis in enumerate(config.axis_order):
        axis_y = 0.88 - index * 0.18
        ax_axes.text(
            0.16,
            axis_y,
            axis,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.3", "fc": config.palette["blue"], "ec": "none"},
            transform=ax_axes.transAxes,
        )
        ax_axes.text(
            0.30,
            axis_y,
            config.axis_labels[axis],
            ha="left",
            va="center",
            transform=ax_axes.transAxes,
        )
    ax_axes.text(
        0.02,
        0.02,
        "multiaxial; never collapsed\nto a diagnostic score",
        transform=ax_axes.transAxes,
        fontsize=6,
        color=config.palette["dark"],
    )
    ax_axes.axis("off")
    add_panel_label(ax_axes, "b", font_size=float(config.export["panel_label_font_pt"]))

    coverage = (
        profiles.assign(biological_participant=_biological_key(profiles))
        .groupby(["dataset_id", "condition"], sort=True, as_index=False)
        .agg(
            n_participants=("biological_participant", "nunique"),
            n_observations=("participant_id", "size"),
            source_artifact_sha256=(
                "source_artifact_sha256",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
            source_table_sha256=(
                "source_table_sha256",
                lambda values: ";".join(sorted(set(values.astype(str)))),
            ),
        )
    )
    coverage["cohort"] = (
        coverage["dataset_id"].astype(str) + " · " + coverage["condition"].astype(str)
    )
    maximum = int(contract.display.get("max_main_cohorts", len(coverage)))
    if maximum < 1:
        raise ValueError("figure_1.display.max_main_cohorts must be positive")
    priority = [str(value) for value in contract.display.get("condition_priority", ())]
    priority_rank = {value: index for index, value in enumerate(priority)}
    coverage["__priority"] = (
        coverage["condition"].astype(str).map(priority_rank).fillna(len(priority))
    )
    coverage = coverage.sort_values(
        ["__priority", "dataset_id", "condition"], kind="mergesort"
    ).drop(columns="__priority")
    displayed_cohorts = set(coverage.head(maximum)["cohort"].astype(str))
    coverage["displayed_in_main_figure"] = coverage["cohort"].astype(str).isin(displayed_cohorts)
    displayed = coverage[coverage["displayed_in_main_figure"]].reset_index(drop=True)
    for position, row in displayed.iterrows():
        n_value = int(row["n_participants"])
        ax_coverage.scatter(
            np.arange(1, n_value + 1),
            np.full(n_value, position),
            s=8,
            color=config.palette["blue"],
            alpha=0.62,
            linewidth=0,
        )
        ax_coverage.text(n_value + 0.5, position, f"n={n_value}", va="center", fontsize=5.8)
    ax_coverage.set_yticks(
        range(len(displayed)), [_short(value, 22) for value in displayed["cohort"].astype(str)]
    )
    ax_coverage.set_xticks([])
    ax_coverage.set_xlabel("each dot = one biological participant")
    ax_coverage.invert_yaxis()
    add_panel_label(ax_coverage, "c", font_size=float(config.export["panel_label_font_pt"]))
    return RenderedFigure(fig, {"c": coverage})


def render_figure_2(
    profiles: pd.DataFrame, config: FigureConfig, contract: FigureContract
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    conditions = _display_values(
        profiles["condition"],
        priority=contract.display.get("condition_priority", ()),
        maximum=int(contract.display.get("max_main_conditions", profiles["condition"].nunique())),
    )
    participant_long = profiles.melt(
        id_vars=[
            "participant_id",
            "dataset_id",
            "condition",
            "source_artifact_sha256",
            "source_table_sha256",
        ],
        value_vars=list(config.axis_order),
        var_name="axis",
        value_name="value",
    )
    all_participants = _participant_values(
        participant_long,
        group_columns=["condition", "axis"],
        value_column="value",
    )
    all_participants["displayed_in_main_figure"] = (
        all_participants["condition"].astype(str).isin(conditions)
    )
    participant_long = all_participants[all_participants["displayed_in_main_figure"]].copy()
    summary = _summary(
        participant_long,
        group_columns=["condition", "axis"],
        value_column="value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    summary = _ordered_summary(summary, column="condition", order=conditions)
    matrix = (
        summary.pivot(index="condition", columns="axis", values="mean")
        .reindex(index=conditions, columns=config.axis_order)
        .to_numpy(dtype=float)
    )
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(2, 5, height_ratios=[0.8, 1.5])
    ax_heat = fig.add_subplot(grid[0, :])
    _heatmap(
        ax_heat,
        matrix,
        row_labels=conditions,
        column_labels=list(config.axis_order),
        palette=config.palette,
        colorbar_label="mean standardized axis",
    )
    add_panel_label(ax_heat, "a", font_size=float(config.export["panel_label_font_pt"]))
    for axis_index, axis in enumerate(config.axis_order):
        ax = fig.add_subplot(grid[1, axis_index])
        part = participant_long.loc[participant_long["axis"] == axis].copy()
        stat = summary.loc[summary["axis"] == axis].copy()
        _dot_interval(
            ax,
            part,
            stat,
            category_column="condition",
            value_column="value",
            palette=config.palette,
            ylabel="standardized axis value" if axis_index == 0 else None,
        )
        ax.set_title(f"{axis} · {config.axis_labels[axis]}", pad=5)
        ax.tick_params(axis="x", rotation=45)
        if axis_index > 0:
            ax.set_ylabel("")
        if axis_index == 0:
            add_panel_label(ax, "b", font_size=float(config.export["panel_label_font_pt"]))
    source = all_participants.merge(summary, on=["condition", "axis"], how="left")
    return RenderedFigure(fig, {"a_b": source})


def render_figure_3(
    content: pd.DataFrame, config: FigureConfig, contract: FigureContract
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    contrasts = _display_values(
        content["contrast"],
        priority=contract.display.get("contrast_priority", ()),
        maximum=int(contract.display.get("max_main_contrasts", content["contrast"].nunique())),
    )
    all_participants = _participant_values(
        content,
        group_columns=["contrast", "axis"],
        value_column="value",
    )
    all_participants["displayed_in_main_figure"] = (
        all_participants["contrast"].astype(str).isin(contrasts)
    )
    participant = all_participants[all_participants["displayed_in_main_figure"]].copy()
    summary = _summary(
        participant,
        group_columns=["contrast", "axis"],
        value_column="value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    summary = _ordered_summary(summary, column="contrast", order=contrasts)
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(2, 5, height_ratios=[1.45, 0.8])
    for axis_index, axis in enumerate(config.axis_order):
        ax = fig.add_subplot(grid[0, axis_index])
        part = participant.loc[participant["axis"] == axis]
        stat = summary.loc[summary["axis"] == axis]
        _dot_interval(
            ax,
            part,
            stat,
            category_column="contrast",
            value_column="value",
            palette=config.palette,
            ylabel="participant contrast effect" if axis_index == 0 else None,
        )
        ax.set_title(f"{axis} · {config.axis_labels[axis]}", pad=5)
        ax.tick_params(axis="x", rotation=45)
        if axis_index == 0:
            add_panel_label(ax, "a", font_size=float(config.export["panel_label_font_pt"]))
    ax_heat = fig.add_subplot(grid[1, :])
    matrix = (
        summary.pivot(index="contrast", columns="axis", values="mean")
        .reindex(index=contrasts, columns=config.axis_order)
        .to_numpy(dtype=float)
    )
    _heatmap(
        ax_heat,
        matrix,
        row_labels=contrasts,
        column_labels=list(config.axis_order),
        palette=config.palette,
        colorbar_label="mean participant effect",
    )
    add_panel_label(ax_heat, "b", font_size=float(config.export["panel_label_font_pt"]))
    source = all_participants.merge(summary, on=["contrast", "axis"], how="left")
    full_effects = content.copy()
    full_effects["displayed_in_main_figure"] = full_effects["contrast"].astype(str).isin(contrasts)
    return RenderedFigure(fig, {"a_b": source, "all_contrast_effects": full_effects})


def render_figure_4(
    participants: pd.DataFrame,
    trajectory: pd.DataFrame,
    config: FigureConfig,
    contract: FigureContract,
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    participant = _participant_values(
        participants.assign(
            # `_participant_values` operates on one value at a time; preserve the
            # second by merging two independently averaged participant tables below.
            _direct=participants["direct_response"]
        ),
        group_columns=["condition"],
        value_column="passive_reachability",
    ).rename(columns={"passive_reachability": "passive_reachability_mean"})
    direct = _participant_values(
        participants,
        group_columns=["condition"],
        value_column="direct_response",
    ).rename(columns={"direct_response": "direct_response_mean"})
    participant = participant.merge(
        direct[
            [
                "condition",
                "dataset_id",
                "participant_id",
                "biological_participant",
                "direct_response_mean",
            ]
        ],
        on=["condition", "dataset_id", "participant_id", "biological_participant"],
        validate="one_to_one",
    )
    delta_passive = _participant_values(
        participants,
        group_columns=["tms_contrast"],
        value_column="passive_delta",
    ).rename(columns={"passive_delta": "passive_delta_mean"})
    delta_direct = _participant_values(
        participants,
        group_columns=["tms_contrast"],
        value_column="direct_delta",
    ).rename(columns={"direct_delta": "direct_delta_mean"})
    delta = delta_passive.merge(
        delta_direct[
            [
                "tms_contrast",
                "dataset_id",
                "participant_id",
                "biological_participant",
                "direct_delta_mean",
            ]
        ],
        on=["tms_contrast", "dataset_id", "participant_id", "biological_participant"],
        validate="one_to_one",
    )
    direct_values = participant.rename(columns={"direct_response_mean": "value"})
    direct_summary = _summary(
        direct_values,
        group_columns=["condition"],
        value_column="value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    trajectory_participant = _participant_values(
        trajectory,
        group_columns=["condition", "time_ms"],
        value_column="trajectory_value",
    )
    trajectory_summary = _summary(
        trajectory_participant,
        group_columns=["condition", "time_ms"],
        value_column="trajectory_value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3])
    ax_scatter = fig.add_subplot(grid[:, 0])
    conditions = sorted(participant["condition"].astype(str).unique())
    colors = _category_colors(conditions, config.palette)
    ax_scatter.scatter(
        delta["passive_delta_mean"],
        delta["direct_delta_mean"],
        s=20,
        color=config.palette["blue"],
        alpha=0.75,
        label=f"paired participants (n={delta['biological_participant'].nunique()})",
    )
    x_values = delta["passive_delta_mean"].to_numpy(dtype=float)
    y_values = delta["direct_delta_mean"].to_numpy(dtype=float)
    if len(x_values) >= 2 and np.ptp(x_values) > 0:
        slope, intercept = np.polyfit(x_values, y_values, deg=1)
        x_line = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        ax_scatter.plot(x_line, intercept + slope * x_line, color=config.palette["dark"], lw=1)
        correlation = float(pd.Series(x_values).corr(pd.Series(y_values), method="spearman"))
    else:
        correlation = float("nan")
    r_text = "undefined" if not np.isfinite(correlation) else f"{correlation:.2f}"
    ax_scatter.text(
        0.03,
        0.97,
        f"Spearman rho={r_text}; n={delta['biological_participant'].nunique()} participants",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
    )
    ax_scatter.set_xlabel("Delta passive reachability\n(awake - propofol)")
    ax_scatter.set_ylabel("Delta direct TMS response\n(awake - propofol)")
    ax_scatter.legend(loc="lower right")
    add_panel_label(ax_scatter, "a", font_size=float(config.export["panel_label_font_pt"]))

    ax_direct = fig.add_subplot(grid[0, 1])
    _dot_interval(
        ax_direct,
        direct_values,
        direct_summary,
        category_column="condition",
        value_column="value",
        palette=config.palette,
        ylabel="direct TMS response",
    )
    add_panel_label(ax_direct, "b", font_size=float(config.export["panel_label_font_pt"]))

    ax_time = fig.add_subplot(grid[1, 1])
    for condition in conditions:
        part = trajectory_summary.loc[
            trajectory_summary["condition"].astype(str) == condition
        ].sort_values("time_ms")
        if part.empty:
            continue
        ax_time.plot(
            part["time_ms"], part["mean"], color=colors[condition], lw=1.2, label=condition
        )
        ax_time.fill_between(
            part["time_ms"].to_numpy(dtype=float),
            part["ci_low"].to_numpy(dtype=float),
            part["ci_high"].to_numpy(dtype=float),
            color=colors[condition],
            alpha=0.18,
            linewidth=0,
        )
    ax_time.axvline(0, color=config.palette["neutral"], linestyle="--", lw=0.8)
    ax_time.set_xlabel("time from pulse (ms)")
    ax_time.set_ylabel("trajectory response")
    ax_time.legend(ncol=min(3, len(conditions)))
    add_panel_label(ax_time, "c", font_size=float(config.export["panel_label_font_pt"]))
    delta_source = delta.copy()
    delta_source["association_test"] = "spearman_participant_level_condition_delta"
    delta_source["spearman_rho"] = correlation
    delta_source["descriptive_line"] = "ordinary_least_squares_no_causal_slope_claim"
    participant_source = participant.merge(
        direct_summary, on="condition", how="left", suffixes=("", "_direct_summary")
    )
    trajectory_source = trajectory_participant.merge(
        trajectory_summary, on=["condition", "time_ms"], how="left"
    )
    return RenderedFigure(
        fig,
        {"a": delta_source, "b": participant_source, "c": trajectory_source},
    )


def render_figure_5(
    robustness: pd.DataFrame,
    fmri: pd.DataFrame | None,
    config: FigureConfig,
    contract: FigureContract,
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    robustness = robustness.copy()
    contrasts = _display_values(
        robustness["contrast"],
        priority=contract.display.get("contrast_priority", ()),
        maximum=int(contract.display.get("max_main_contrasts", 1)),
    )
    display_analysis = {
        "covariance_dwell_matched_state_space": "state-space null",
        "post_encoder_latent_rotation_control": "latent rotation",
        "blockwise_temporal_permutation": "block permutation",
        "phase_randomization": "phase randomization",
    }
    robustness["robustness_check"] = (
        robustness["analysis"].astype(str).replace(display_analysis)
        + " · "
        + robustness["metric"].astype(str)
    )
    candidates = robustness[robustness["contrast"].astype(str).isin(contrasts)].copy()
    analysis_priority = {
        str(value): index
        for index, value in enumerate(contract.display.get("analysis_priority", ()))
    }
    metric_priority = {
        str(value): index for index, value in enumerate(contract.display.get("metric_priority", ()))
    }
    checks = candidates[["analysis", "metric", "robustness_check"]].drop_duplicates()
    checks["__analysis_order"] = (
        checks["analysis"].astype(str).map(analysis_priority).fillna(len(analysis_priority))
    )
    checks["__metric_order"] = (
        checks["metric"].astype(str).map(metric_priority).fillna(len(metric_priority))
    )
    checks = checks.sort_values(
        ["__analysis_order", "__metric_order", "robustness_check"], kind="mergesort"
    )
    maximum_checks = int(contract.display.get("max_main_checks", len(checks)))
    if maximum_checks < 1:
        raise ValueError("figure_5.display.max_main_checks must be positive")
    displayed_checks = checks.head(maximum_checks)["robustness_check"].astype(str).tolist()
    robustness["displayed_in_main_figure"] = robustness["contrast"].astype(str).isin(
        contrasts
    ) & robustness["robustness_check"].astype(str).isin(displayed_checks)
    displayed = robustness[robustness["displayed_in_main_figure"]].copy()
    participant = _participant_values(
        displayed,
        group_columns=["robustness_check"],
        value_column="value",
    )
    summary = _summary(
        participant,
        group_columns=["robustness_check"],
        value_column="value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    summary = _ordered_summary(summary, column="robustness_check", order=displayed_checks)
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax_robust = fig.add_subplot(grid[0, 0])
    _dot_interval(
        ax_robust,
        participant,
        summary,
        category_column="robustness_check",
        value_column="value",
        palette=config.palette,
        ylabel="signed observed-null effect survival",
        horizontal=True,
    )
    add_panel_label(ax_robust, "a", font_size=float(config.export["panel_label_font_pt"]))

    source_panels: dict[str, pd.DataFrame] = {
        "a": participant.merge(summary, on="robustness_check", how="left"),
        # Keep every repeat and every non-displayed contrast unchanged.  The
        # participant summary above is a separate table so it cannot be mistaken
        # for repeat-level provenance.
        "all_null_effects": robustness,
    }
    if fmri is None:
        ax_fmri = fig.add_subplot(grid[0, 1])
        ax_fmri.text(
            0.5,
            0.55,
            "fMRI triangulation\nnot supplied in this run",
            ha="center",
            va="center",
            transform=ax_fmri.transAxes,
            color=config.palette["neutral"],
        )
        ax_fmri.text(
            0.5,
            0.38,
            "no simulated replacement",
            ha="center",
            va="center",
            transform=ax_fmri.transAxes,
            fontsize=6,
            color=config.palette["dark"],
        )
        ax_fmri.axis("off")
        add_panel_label(ax_fmri, "b", font_size=float(config.export["panel_label_font_pt"]))
    else:
        axes = [axis for axis in config.axis_order if axis != "P"]
        participant_fmri = fmri.melt(
            id_vars=[
                "participant_id",
                "dataset_id",
                "condition",
                "source_artifact_sha256",
                "source_table_sha256",
            ],
            value_vars=axes,
            var_name="axis",
            value_name="value",
        )
        participant_fmri = _participant_values(
            participant_fmri,
            group_columns=["condition", "axis"],
            value_column="value",
        )
        summary_fmri = _summary(
            participant_fmri,
            group_columns=["condition", "axis"],
            value_column="value",
            repetitions=int(config.integrity["bootstrap_repetitions"]),
        )
        fmri_grid = grid[0, 1].subgridspec(2, 2)
        for axis_index, axis in enumerate(axes):
            ax_fmri = fig.add_subplot(fmri_grid[axis_index // 2, axis_index % 2])
            part = participant_fmri.loc[participant_fmri["axis"] == axis]
            stat = summary_fmri.loc[summary_fmri["axis"] == axis]
            _dot_interval(
                ax_fmri,
                part,
                stat,
                category_column="condition",
                value_column="value",
                palette=config.palette,
                ylabel="fMRI-compatible axis" if axis_index % 2 == 0 else None,
            )
            ax_fmri.set_title(f"{axis} · {config.axis_labels[axis]}", pad=5)
            ax_fmri.tick_params(axis="x", rotation=45)
            if axis_index == 0:
                add_panel_label(ax_fmri, "b", font_size=float(config.export["panel_label_font_pt"]))
            if axis_index == len(axes) - 1:
                ax_fmri.text(
                    1.0,
                    -0.42,
                    "passive triangulation; P excluded",
                    transform=ax_fmri.transAxes,
                    ha="right",
                    fontsize=5.5,
                    color=config.palette["neutral"],
                )
        source_panels["b"] = participant_fmri.merge(
            summary_fmri, on=["condition", "axis"], how="left"
        )
    return RenderedFigure(fig, source_panels)


def render_figure_6(
    clinical: pd.DataFrame, config: FigureConfig, contract: FigureContract
) -> RenderedFigure:
    plt = _pyplot()
    apply_publication_style(float(config.export["body_font_pt"]))
    clinical = clinical.copy()
    diagnosis = clinical["diagnosis"].astype("string").str.strip()
    clinical["diagnosis_display"] = diagnosis.where(
        diagnosis.notna() & diagnosis.ne(""), "diagnosis unavailable"
    ).astype(str)
    clinical_long = clinical.melt(
        id_vars=[
            "participant_id",
            "dataset_id",
            "diagnosis",
            "diagnosis_display",
            "crs_r_total",
            WAKE_REGIME_LLR,
            "source_artifact_sha256",
            "source_table_sha256",
        ],
        value_vars=list(config.axis_order),
        var_name="axis",
        value_name="value",
    )
    all_participants = _participant_values(
        clinical_long,
        group_columns=["diagnosis_display", "axis"],
        value_column="value",
    )
    participant_diagnosis = clinical[
        ["dataset_id", "participant_id", "diagnosis", "diagnosis_display"]
    ].drop_duplicates()
    all_participants = all_participants.merge(
        participant_diagnosis,
        on=["dataset_id", "participant_id", "diagnosis_display"],
        how="left",
        validate="many_to_one",
    )
    diagnoses = _display_values(
        clinical["diagnosis_display"],
        priority=contract.display.get("diagnosis_priority", ()),
        maximum=int(
            contract.display.get("max_main_diagnoses", clinical["diagnosis_display"].nunique())
        ),
    )
    all_participants["displayed_in_main_figure"] = (
        all_participants["diagnosis_display"].astype(str).isin(diagnoses)
    )
    participant = all_participants[all_participants["displayed_in_main_figure"]].copy()
    summary = _summary(
        participant,
        group_columns=["diagnosis_display", "axis"],
        value_column="value",
        repetitions=int(config.integrity["bootstrap_repetitions"]),
    )
    summary = _ordered_summary(summary, column="diagnosis_display", order=diagnoses)
    fig = plt.figure(figsize=figure_size(contract.final_size_mm), layout="constrained")
    grid = fig.add_gridspec(2, 10, height_ratios=[1, 1.15])
    for axis_index, axis in enumerate(config.axis_order):
        ax = fig.add_subplot(grid[0, axis_index * 2 : (axis_index + 1) * 2])
        part = participant.loc[participant["axis"] == axis]
        stat = summary.loc[summary["axis"] == axis]
        _dot_interval(
            ax,
            part,
            stat,
            category_column="diagnosis_display",
            value_column="value",
            palette=config.palette,
            ylabel="frozen profile axis" if axis_index == 0 else None,
        )
        ax.set_title(f"{axis} · {config.axis_labels[axis]}", pad=5)
        ax.tick_params(axis="x", rotation=45)
        if axis_index == 0:
            add_panel_label(ax, "a", font_size=float(config.export["panel_label_font_pt"]))

    ax_crs = fig.add_subplot(grid[1, :5])
    clinical_participant = (
        clinical.groupby(
            ["dataset_id", "participant_id", "diagnosis_display"],
            as_index=False,
            sort=True,
            dropna=False,
        )[["crs_r_total", WAKE_REGIME_LLR]]
        .mean()
        .reset_index(drop=True)
    )
    clinical_participant = clinical_participant.merge(
        participant_diagnosis,
        on=["dataset_id", "participant_id", "diagnosis_display"],
        how="left",
        validate="one_to_one",
    )
    clinical_participant["displayed_in_main_figure"] = (
        clinical_participant["diagnosis_display"].astype(str).isin(diagnoses)
    )
    finite_endpoint = np.isfinite(
        clinical_participant["crs_r_total"].to_numpy(dtype=float)
    ) & np.isfinite(clinical_participant[WAKE_REGIME_LLR].to_numpy(dtype=float))
    clinical_participant["crs_r_available"] = finite_endpoint
    colors = _category_colors(diagnoses, config.palette)
    for diagnosis in diagnoses:
        part = clinical_participant.loc[
            clinical_participant["diagnosis_display"].astype(str).eq(diagnosis)
            & clinical_participant["crs_r_available"]
        ]
        if part.empty:
            continue
        ax_crs.scatter(
            part["crs_r_total"],
            part[WAKE_REGIME_LLR],
            s=18,
            color=colors[diagnosis],
            alpha=0.7,
            label=f"{diagnosis} (n={len(part)})",
        )
    plotted_endpoint = clinical_participant.loc[
        clinical_participant["displayed_in_main_figure"] & clinical_participant["crs_r_available"]
    ]
    crs = plotted_endpoint["crs_r_total"].to_numpy(dtype=float)
    preservation = plotted_endpoint[WAKE_REGIME_LLR].to_numpy(dtype=float)
    if len(crs) >= 2 and np.ptp(crs) > 0:
        slope, intercept = np.polyfit(crs, preservation, deg=1)
        x_line = np.linspace(float(crs.min()), float(crs.max()), 100)
        ax_crs.plot(x_line, intercept + slope * x_line, color=config.palette["dark"], lw=1)
        correlation = (
            float(stats.spearmanr(crs, preservation).statistic)
            if np.ptp(preservation) > 0
            else float("nan")
        )
    else:
        correlation = float("nan")
    if len(crs) == 0:
        endpoint_text = "CRS-R unavailable in supplied locked profiles"
    elif len(crs) == 1:
        endpoint_text = "CRS-R supplied for one participant; association not estimated"
    else:
        r_text = "undefined" if not np.isfinite(correlation) else f"{correlation:.2f}"
        endpoint_text = f"Spearman rho={r_text}; n={len(crs)}"
    ax_crs.text(
        0.03,
        0.97,
        endpoint_text,
        transform=ax_crs.transAxes,
        ha="left",
        va="top",
        color=config.palette["neutral"] if len(crs) < 2 else config.palette["dark"],
    )
    ax_crs.set_xlabel("CRS-R total")
    ax_crs.set_ylabel("wake vs propofol log-likelihood ratio")
    if ax_crs.get_legend_handles_labels()[0]:
        ax_crs.legend(loc="lower right")
    add_panel_label(ax_crs, "b", font_size=float(config.export["panel_label_font_pt"]))

    ax_heterogeneity = fig.add_subplot(grid[1, 5:])
    clinical_indexed = clinical.copy().sort_values(
        [WAKE_REGIME_LLR, "dataset_id", "participant_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    matrix = clinical_indexed[list(config.axis_order)].to_numpy(dtype=float)
    _heatmap(
        ax_heterogeneity,
        matrix,
        row_labels=["" for _ in range(len(clinical_indexed))],
        column_labels=list(config.axis_order),
        palette=config.palette,
        colorbar_label="frozen axis value",
    )
    ax_heterogeneity.set_ylabel(f"all participants (n={len(clinical_indexed)}; IDs hidden)")
    add_panel_label(ax_heterogeneity, "c", font_size=float(config.export["panel_label_font_pt"]))
    source_a = all_participants.merge(summary, on=["diagnosis_display", "axis"], how="left")
    source_b = clinical_participant.merge(
        clinical[
            ["dataset_id", "participant_id", "source_artifact_sha256", "source_table_sha256"]
        ].drop_duplicates(),
        on=["dataset_id", "participant_id"],
        how="left",
        validate="one_to_one",
    )
    source_b["association_test"] = "spearman_participant_level"
    source_b["spearman_rho"] = correlation
    source_b["n_association"] = len(crs)
    source_b["trend_line"] = "descriptive_least_squares_no_slope_inference"
    source_c = clinical_indexed.copy()
    source_c.insert(0, "display_order", np.arange(1, len(source_c) + 1))
    return RenderedFigure(fig, {"a": source_a, "b": source_b, "c": source_c})

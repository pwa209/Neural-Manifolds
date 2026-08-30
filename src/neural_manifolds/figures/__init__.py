"""Python-only, source-data-grounded manuscript figure production."""

from .config import FigureConfig, FigureContract, load_figure_config
from .io import FigureInputError
from .stage import (
    FigureRunResult,
    figure_run_artifacts,
    run_clinical_figure_supplement,
    run_figures,
    run_fmri_figure_supplement,
)

__all__ = [
    "FigureConfig",
    "FigureContract",
    "FigureInputError",
    "FigureRunResult",
    "figure_run_artifacts",
    "load_figure_config",
    "run_clinical_figure_supplement",
    "run_figures",
    "run_fmri_figure_supplement",
]

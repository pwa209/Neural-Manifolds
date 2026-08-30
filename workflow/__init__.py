"""Restartable phase orchestration for the Neural Manifolds study.

The workflow package is deliberately independent of a cluster scheduler.  It is
invoked inside a durable ``tmux`` session on the university server and delegates
scientific work to the repository's ``neural-manifolds run-phase`` command.
"""

from .phases import PHASES, PhaseSpec

__all__ = ["PHASES", "PhaseSpec"]

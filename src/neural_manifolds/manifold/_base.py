"""Small estimator utilities with the subset of the scikit-learn protocol we use.

The project deliberately keeps the numerical core dependent only on NumPy.  The
estimators nevertheless implement ``get_params``/``set_params`` and use explicit
constructor arguments, which makes them cloneable and usable in scikit-learn
model-selection code when scikit-learn is installed.
"""

from __future__ import annotations

import inspect
from typing import Any


class EstimatorMixin:
    """Implement the parameter introspection part of the sklearn estimator API."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters.

        ``deep`` is accepted for API compatibility.  Nested estimators are not
        expanded because the manifold estimators currently contain only scalar
        and tuple-valued configuration.
        """

        del deep
        signature = inspect.signature(self.__init__)
        parameters: dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if not hasattr(self, name):
                raise AttributeError(
                    f"{type(self).__name__}.__init__ must assign parameter {name!r} "
                    "to an attribute of the same name"
                )
            parameters[name] = getattr(self, name)
        return parameters

    def set_params(self, **params: Any) -> EstimatorMixin:
        """Set constructor parameters, rejecting misspelled names."""

        valid = self.get_params(deep=False)
        unknown = sorted(set(params).difference(valid))
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) {unknown!r} for {type(self).__name__}; "
                f"valid parameters are {sorted(valid)!r}"
            )
        for name, value in params.items():
            setattr(self, name, value)
        return self


def require_fitted(estimator: object, *attributes: str) -> None:
    """Raise a helpful error if an estimator has not been fitted."""

    missing = [name for name in attributes if not hasattr(estimator, name)]
    if missing:
        raise RuntimeError(
            f"{type(estimator).__name__} is not fitted; missing attributes {', '.join(missing)}"
        )

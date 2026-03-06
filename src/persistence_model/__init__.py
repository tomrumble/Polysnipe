"""Persistence model package.

This package provides a compact, well-documented implementation of the
state-persistence calculation described in ``architecture.md``.
"""

from .model import PersistenceInputs, PersistenceModel, PersistenceOutput

__all__ = ["PersistenceInputs", "PersistenceModel", "PersistenceOutput"]

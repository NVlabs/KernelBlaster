"""Durable local state for resumable KernelBlaster runs.

The control process owns this package.  GPU workers interact with it through
the control API rather than opening the SQLite database themselves.
"""

from .cas import ContentAddressedStore
from .repository import JobRepository
from .state import StateStore, resolve_state_paths, state_storage_requested

__all__ = [
    "ContentAddressedStore",
    "JobRepository",
    "StateStore",
    "resolve_state_paths",
    "state_storage_requested",
]

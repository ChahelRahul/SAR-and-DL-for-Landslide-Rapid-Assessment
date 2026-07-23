"""SAR-LRA operational package scaffold.

Scientific processing remains in the versioned notebooks until it is extracted
and tested in the next implementation issue.
"""

from .constants import MODEL_NAME, MODEL_VERSION, EXPECTED_BAND_ORDER

__all__ = ["MODEL_NAME", "MODEL_VERSION", "EXPECTED_BAND_ORDER"]

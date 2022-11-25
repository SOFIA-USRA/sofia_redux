# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""SOFIA specific exceptions."""

__all__ = ['SOFIAImportError']


class SOFIAImportError(ImportError):
    """Raised when a SOFIA instrument is not available"""
    pass

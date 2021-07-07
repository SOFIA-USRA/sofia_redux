# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base class for pipecal errors."""

__all__ = ['PipeCalError']


class PipeCalError(ValueError):
    """A ValueError raised by pipecal functions."""
    pass

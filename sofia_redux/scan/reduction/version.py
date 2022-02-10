# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy.time import Time

__all__ = ['ReductionVersion']


class ReductionVersion(ABC):

    version = '1.0.0'
    revision = ''

    def __init__(self):
        self.home = '.'
        self.work_path = '.'

    @classmethod
    def add_history(cls, header):
        """
        Add history messages to a header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        t_now = Time(Time.now(), format='isot').value
        version = cls.get_full_version()
        header['HISTORY'] = f'Reduced: SOFSCAN v{version} @ {t_now}'

    @classmethod
    def get_full_version(cls):
        """
        Return the SOFSCAN full version string.

        Returns
        -------
        str
        """
        if cls.revision is None:
            return cls.version
        elif len(cls.revision) == 0:
            return cls.version
        else:
            return f'{cls.version} ({cls.revision})'


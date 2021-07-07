# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import TestCase
import warnings

from astropy.io import fits
import numpy as np


def add_jailbars(data, level=10):
    if len(data.shape) == 3:
        for frame in data:
            for i in range((frame.shape[1] // 16)):
                frame[:, i * 16] += level
    else:
        for i in range((data.shape[1] // 16)):
            data[:, i * 16] += level


def random_mask(shape, frac=0.5):
    mask = np.full(np.product(shape), False)
    mask[:int(np.product(shape) * frac)] = True
    np.random.shuffle(mask)
    return np.reshape(mask, shape)


def run_command(command, stdout=False, env=None, cwd=None):
    if env is None:
        env = os.environ.copy()

    if stdout:
        process = subprocess.Popen(command, shell=True, env=env, cwd=cwd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        text, _ = process.communicate()
        text = text.decode().strip()
        exitcode = process.returncode
        return exitcode, text
    else:
        process = subprocess.Popen(command, shell=True, env=env, cwd=cwd)
        exitcode = process.wait()
        return exitcode


def make_test_fits(filepath, data=None, keywords=None):
    """
    Create a mock FITS file

    Parameters
    ----------
    filepath : str
        path to FITS file
    data : numpy.ndarray
        alternative to numpy.arange(100)
    keywords : dict
        keyword values to add to header

    Returns
    -------
    None
    """
    if data is None:
        data = np.arange(100)
    hdu = fits.PrimaryHDU(data)
    for keyword, value in keywords.items():
        hdu.header[keyword] = value
    hdu.writeto(filepath)


class CheckOutput(object):
    def __init__(self):
        self.data = []

    def write(self, msg):
        self.data.append(msg)

    def __str__(self):
        return ''.join(self.data)


class PipelineTestCase(TestCase):
    def setUp(self):
        """Unittest does not seem to recognize built code, hence the ignore
        warnings"""
        ignore_warnings = [
            "numpy.dtype size changed, may indicate binary incompatibility",
            "can't resolve package from __spec__ or __package__",
            "the imp module is deprecated in favour of importlib"
        ]
        for phrase in ignore_warnings:
            warnings.filterwarnings("ignore", message=phrase)
        self.this_file = os.path.realpath(
            sys.modules[self.__class__.__module__].__file__)
        self.data_dir = os.path.join(os.path.dirname(self.this_file), 'data')
        self.tempdir = tempfile.mkdtemp(prefix=self.__module__ + '-')
        _, self.tempfile = tempfile.mkstemp(dir=self.tempdir)

    def clear_file(self):
        if os.path.isfile(self.tempfile):
            os.remove(self.tempfile)
        with open(self.tempfile, 'w') as _:
            pass

    def file_contents(self):
        with open(self.tempfile, 'r') as f:
            contents = f.read()
        return contents

    def tearDown(self):
        for attr, action in [('tempfile', os.remove),
                             ('tempdir', shutil.rmtree)]:
            obj = getattr(self, attr, None)
            if not isinstance(obj, str):
                continue
            for _ in range(5):
                try:
                    action(obj)
                except (OSError, IOError):
                    time.sleep(0.5)

    def fake_fits(self, data=None, keywords=None):
        if data is None:
            data = np.arange(100)
        hdu = fits.PrimaryHDU(data)
        if keywords is not None:
            for keyword, value in keywords.items():
                hdu.header[keyword] = value
        if os.path.isfile(self.tempfile):
            os.remove(self.tempfile)
        hdu.writeto(self.tempfile)
        return self.tempfile

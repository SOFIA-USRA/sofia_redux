# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.integration.dependents.dependents import Dependents


class TestDependents(object):

    def test_init(self, populated_integration):
        # minimal init requires populated integration
        integ = populated_integration
        dep = Dependents(integ, 'test')

        assert dep.name == 'test'
        assert dep.integration is integ
        assert dep.for_frame.size == 1100
        assert np.all(dep.for_frame == 0)
        assert dep.for_channel.size == 121
        assert np.all(dep.for_channel == 0)
        assert integ.dependents['test'] is dep

        dep2 = dep.copy()
        assert dep2 is not dep
        assert dep2.name == 'test'
        assert dep2.integration is integ
        assert dep2.for_frame.size == 1100
        assert np.all(dep2.for_frame == 0)
        assert dep2.for_channel.size == 121
        assert np.all(dep2.for_channel == 0)
        assert integ.dependents['test'] is dep2

    def test_async(self, populated_integration):
        integ = populated_integration
        dep = Dependents(integ, 'test')

        dep.add_async(integ.channels, 2.0)
        dep.add_async(integ.frames, 3.0)
        assert np.allclose(dep.for_channel, 2)
        assert np.allclose(dep.for_frame, 3)

        # error if not channels or frames instance
        with pytest.raises(ValueError) as err:
            dep.add_async(integ.frames.data, 1.0)
        assert 'Must be <class' in str(err)

    def test_add_for_channels(self, populated_integration):
        dep = Dependents(populated_integration, 'test')
        dep.add_for_channels(3)
        assert np.allclose(dep.for_channel, 3)

        arr = np.arange(dep.for_channel.size)
        dep.add_for_channels(arr)
        assert np.allclose(dep.for_channel, arr + 3)

    def test_add_for_frames(self, populated_integration):
        dep = Dependents(populated_integration, 'test')
        dep.add_for_frames(3)
        assert np.allclose(dep.for_frame, 3)

        arr = np.arange(dep.for_frame.size)
        dep.add_for_frames(arr)
        assert np.allclose(dep.for_frame, arr + 3)

    def test_clear(self, populated_integration, mocker):
        integ = populated_integration
        dep = Dependents(integ, 'test')
        dep.add_for_channels(3)
        dep.add_for_frames(3)

        m1 = mocker.patch.object(integ.channels, 'remove_dependents')
        m2 = mocker.patch.object(integ.frames, 'remove_dependents')

        dep.clear()
        assert np.all(dep.for_channel == 0)
        assert np.all(dep.for_frame == 0)
        m1.assert_called_once()
        m2.assert_called_once()

    def test_apply(self, populated_integration, mocker):
        integ = populated_integration
        dep = Dependents(integ, 'test')
        dep.add_for_channels(3)
        dep.add_for_frames(3)

        m1 = mocker.patch.object(integ.channels, 'add_dependents')
        m2 = mocker.patch.object(integ.frames, 'add_dependents')

        # without channel indices
        dep.apply()
        assert np.all(dep.for_channel == 3)
        assert np.all(dep.for_frame == 3)
        m1.assert_called_once()
        m2.assert_called_once()

        # with indices
        integ.channels.indices = [1, 2, 3]
        dep.apply()
        m1.assert_called()

    def test_get(self, populated_integration):
        integ = populated_integration
        dep = Dependents(integ, 'test')

        # error for inappropriate type
        with pytest.raises(ValueError) as err:
            dep.get(integ)
        assert 'Must be <class' in str(err)

        # if ChannelData or Frames, return for_channel or for_frame
        cdata = dep.get(integ.channels.data)
        assert np.allclose(cdata, dep.for_channel)
        fdata = dep.get(integ.frames)
        assert np.allclose(fdata, dep.for_frame)

        # if other type but has data attribute, get data from that
        cdata = dep.get(integ.channels)
        assert np.allclose(cdata, dep.for_channel)

        # with indices
        integ.channels.data.indices = [1, 2, 3]
        cdata = dep.get(integ.channels)
        assert np.allclose(cdata, dep.for_channel[1:4])

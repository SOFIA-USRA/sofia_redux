# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.camera.camera import Camera
from sofia_redux.scan.channels.channels import Channels
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.toolkit.utilities.fits import set_log_level


@pytest.fixture
def populated_camera(populated_data):
    camera = Camera(info=ExampleInfo())
    camera.data = populated_data
    camera.initialize()
    return camera


@pytest.fixture
def rcp_file(tmpdir):
    rcp = tmpdir.join('test.rcp')
    example = ("  9  0.955  1.101   527.03   264.71\n"
               " 10  1.325  0.857   557.01   324.56\n")
    rcp.write(example)
    return str(rcp)


@pytest.fixture
def rcp_file_3col(tmpdir):
    rcp = tmpdir.join('test3col.rcp')
    example = ("  9  527.03   264.71\n"
               " 10  557.01   324.56\n")
    rcp.write(example)
    return str(rcp)


@pytest.fixture
def rcp_file_4col(tmpdir):
    rcp = tmpdir.join('test4col.rcp')
    example = ("  9  0.955   527.03   264.71\n"
               " 10  1.325   557.01   324.56\n")
    rcp.write(example)
    return str(rcp)


class TestCamera(object):

    def test_init(self):
        camera = Camera()
        assert isinstance(camera, Channels)

    def test_rotation(self, populated_camera):
        camera = Camera()
        assert camera.rotation is None
        camera.rotation = 5 * units.deg
        assert camera.rotation is None
        assert camera.get_rotation_angle() is None

        assert populated_camera.rotation == 0 * units.deg
        populated_camera.rotation = 5 * units.deg
        assert populated_camera.rotation == 5 * units.deg
        assert populated_camera.get_rotation_angle() == 5 * units.deg

    def test_init_modalities(self, populated_camera):
        populated_camera.init_modalities()
        assert 'gradients' in populated_camera.modalities
        grad = populated_camera.modalities['gradients'].modes
        assert len(grad) == 2
        assert grad[0].name == 'gradients:x'
        assert grad[1].name == 'gradients:y'

    def test_set_reference_position(self, populated_camera):
        pix = populated_camera.data.position.copy()
        ref = Coordinate2D([0, 0])
        populated_camera.set_reference_position(ref)
        assert np.allclose(populated_camera.data.position.x, pix.x)
        assert np.allclose(populated_camera.data.position.y, pix.y)

        ref = Coordinate2D([1, 2], unit=units.arcsec)
        populated_camera.set_reference_position(ref)
        assert np.allclose(populated_camera.data.position.x,
                           pix.x - 1 * units.arcsec)
        assert np.allclose(populated_camera.data.position.y,
                           pix.y - 2 * units.arcsec)

    @pytest.mark.parametrize('cols', [3, 4, 5])
    def test_load_channel_data(self, tmpdir, capsys, mocker,
                               populated_camera, rcp_file,
                               rcp_file_4col, rcp_file_3col, cols):
        populated_camera.configuration.set_option('rotation', 5)
        populated_camera.load_channel_data()
        assert populated_camera.rotation == 5 * units.deg
        populated_camera.configuration.set_option('rotation', 0)

        # assign an rcp file that doesn't exist
        populated_camera.configuration.set_option('rcp', 'bad.rcp')
        populated_camera.load_channel_data()
        assert 'Cannot update pixel RCP data' in capsys.readouterr().err

        # assign one that exists but has invalid columns
        rcp = tmpdir.join('invalid.rcp')
        rcp.write('# test\n1')
        populated_camera.configuration.set_option('rcp', str(rcp))
        populated_camera.load_channel_data()
        assert 'Invalid number of columns' in capsys.readouterr().err

        # assign one with proper format
        px = populated_camera.data.position.x.copy()
        py = populated_camera.data.position.y.copy()
        gain = populated_camera.data.gain.copy()
        coupling = populated_camera.data.coupling.copy()

        if cols == 3:
            rcp_test = rcp_file_3col
        elif cols == 4:
            rcp_test = rcp_file_4col
        else:
            rcp_test = rcp_file
        populated_camera.configuration.set_option('rcp', rcp_test)
        populated_camera.load_channel_data()
        assert 'Cannot update' not in capsys.readouterr().err

        # rcp positions directly overridden by file
        pos = populated_camera.data.position
        assert pos.x[9] == 527.03 * units.arcsec
        assert pos.y[9] == 264.71 * units.arcsec
        assert pos.x[10] == 557.01 * units.arcsec
        assert pos.y[10] == 324.56 * units.arcsec
        # other positions unmodified
        assert np.allclose(pos.x[:9], px[:9])
        assert np.allclose(pos.x[11:], px[11:])
        assert np.allclose(pos.y[:9], py[:9])
        assert np.allclose(pos.y[11:], py[11:])
        # gain and coupling not used
        assert np.allclose(populated_camera.data.gain, gain)
        assert np.allclose(populated_camera.data.coupling, coupling)

        # use rcp gains and set rcp center, rotate, and scale
        populated_camera.configuration.set_option('rcp.gains', True)
        populated_camera.configuration.set_option('rcp.center', [1, 1])
        populated_camera.configuration.set_option('rcp.rotate', 90)
        populated_camera.configuration.set_option('rcp.zoom', 2)
        populated_camera.load_channel_data()
        pos = populated_camera.data.position

        # rcp data overridden by file and modified
        assert np.isclose(pos.x[9],
                          -527.42 * units.arcsec)
        assert np.isclose(pos.y[9],
                          1052.06 * units.arcsec)
        assert np.isclose(populated_camera.data.gain[9], 1)
        if cols == 3:
            assert np.isclose(populated_camera.data.coupling[9], coupling[9])
        elif cols == 4:
            assert np.isclose(populated_camera.data.coupling[9], 0.955)
        else:
            assert np.isclose(populated_camera.data.coupling[9], 0.955 / 1.101)

        assert np.isclose(pos.x[10],
                          -647.12 * units.arcsec)
        assert np.isclose(pos.y[10],
                          1112.02 * units.arcsec)
        assert np.isclose(populated_camera.data.gain[10], 1)
        if cols == 3:
            assert np.isclose(populated_camera.data.coupling[10], coupling[10])
        elif cols == 4:
            assert np.isclose(populated_camera.data.coupling[10], 1.325)
        else:
            assert np.isclose(populated_camera.data.coupling[10],
                              1.325 / 0.857)

        # other pixel positions modified by scale and rotate
        assert not np.allclose(pos.x[:9], px[:9])
        assert not np.allclose(pos.y[:9], py[:9])

        # gain and coupling not modified
        assert np.allclose(populated_camera.data.gain[:9], gain[:9])
        assert np.allclose(populated_camera.data.gain[11:], gain[11:])
        assert np.allclose(populated_camera.data.coupling[:9], coupling[:9])
        assert np.allclose(populated_camera.data.coupling[11:], coupling[11:])

    def test_get_none_rcp(self):
        assert Camera.get_rcp_info(None) is None

    def test_get_rcp_header(self):
        camera = Camera()
        expected = "ch\t[Gpnt]\t[Gsky]ch\t[dX\"]\t[dY\"]"
        assert camera.get_rcp_header() == expected

    def test_print_pixel_rcp(self, populated_camera):
        rcp = populated_camera.print_pixel_rcp(header='# test')
        assert rcp.startswith('# SOFSCAN')
        assert '# test\n' in rcp
        assert rcp.strip().endswith('120\t1.000\t1.000\t-1.000e+01\t-1.000e+01')
        assert len(rcp.split('\n')) == 127

    def test_rotate(self, capsys, populated_camera):
        with set_log_level('DEBUG'):
            populated_camera.rotate(np.nan)
        assert 'Applying' not in capsys.readouterr().out
        assert populated_camera.rotation == 0 * units.deg
        px = populated_camera.data.position.x.copy()
        py = populated_camera.data.position.y.copy()

        with set_log_level('DEBUG'):
            populated_camera.rotate(90 * units.deg)
        assert 'Applying' in capsys.readouterr().out
        assert populated_camera.rotation == 90 * units.deg
        assert np.allclose(populated_camera.data.position.x, -py)
        assert np.allclose(populated_camera.data.position.y, px)

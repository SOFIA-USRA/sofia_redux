# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import matplotlib as mpl
import numpy as np
import pytest

from sofia_redux.visualization import quicklook


class TestQuicklookImage(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_hdulist_image(self, tmpdir, simple_fits_data,
                           figures_same):
        hdul = simple_fits_data

        fig = quicklook.make_image(hdul)
        assert isinstance(fig, mpl.figure.Figure)

        # 2 axes - plot and colorbar
        assert len(fig.get_axes()) == 2

        # missing celestial WCS - will label with x and y
        assert fig.get_axes()[0].get_xlabel() == 'x'
        assert fig.get_axes()[0].get_ylabel() == 'y'

        # write to file on disk; output figure should be the same
        fname = str(tmpdir.join('test.fits'))
        hdul.writeto(fname)
        hdul.close()

        fig2 = quicklook.make_image(fname)
        assert isinstance(fig2, mpl.figure.Figure)

        assert figures_same(fig, fig2)

    def test_bad_input(self, capsys):
        badfile = 'bad_input.fits'
        with pytest.raises(ValueError) as err:
            quicklook.make_image(badfile)
        assert 'No input file' in str(err)

    def test_bad_extension(self, capsys, simple_fits_data):
        with pytest.raises(ValueError) as err:
            quicklook.make_image(simple_fits_data, extension=2)
        assert 'No extension 2 present' in str(err)

    def test_title(self, simple_fits_data):
        hdul = simple_fits_data

        # test title
        title = 'Test title'
        fig = quicklook.make_image(hdul, title=title)
        ax = fig.get_axes()[0]
        assert fig._suptitle.get_text() == title
        assert ax.title.get_text() == ''

        # add all combinations of subtitles too
        subtitle = 'Test subtitle'
        subsubtitle = 'Test subsubtitle'

        fig = quicklook.make_image(hdul, title=title,
                                   subtitle=subtitle,
                                   subsubtitle=subsubtitle)
        ax = fig.get_axes()[0]
        assert fig._suptitle.get_text() == title
        assert ax.title.get_text() == '\n'.join([subtitle, subsubtitle])

        fig = quicklook.make_image(hdul, title=title,
                                   subtitle=subtitle)
        ax = fig.get_axes()[0]
        assert fig._suptitle.get_text() == title
        assert ax.title.get_text() == subtitle

        fig = quicklook.make_image(hdul, title=title,
                                   subsubtitle=subsubtitle)
        ax = fig.get_axes()[0]
        assert fig._suptitle.get_text() == title
        assert ax.title.get_text() == subsubtitle

        hdul.close()

    def test_scale(self, simple_fits_data):
        hdul = simple_fits_data

        # default: 0.25, 99.75, on value range 0 - 99
        fig = quicklook.make_image(hdul, scale=None)
        ax = fig.get_axes()[0]
        assert np.allclose(ax.images[-1].get_clim(), [.25, 99.75], atol=1)

        # override default
        fig = quicklook.make_image(hdul, scale=[0, 80])
        ax = fig.get_axes()[0]
        assert np.allclose(ax.images[-1].get_clim(), [0, 80], atol=1)

        hdul.close()

    def test_wcs(self, capsys, wcs_fits_data):
        hdul = wcs_fits_data

        fig = quicklook.make_image(hdul, crop_region=None)
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [0, 9], atol=0.5)
        assert np.allclose(ax.get_ylim(), [0, 9], atol=0.5)

        assert 'No celestial WCS' not in capsys.readouterr().err
        assert ax.get_xlabel() == 'RA (J2000)'
        assert ax.get_ylabel() == 'Dec (J2000)'

        fig = quicklook.make_image(hdul, crop_region=[5, 7, 4, 6])
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [3, 7], atol=0.1)
        assert np.allclose(ax.get_ylim(), [4, 10], atol=0.1)

        # check with pixel crop units
        fig = quicklook.make_image(hdul, crop_region=[5, 7, 4, 6],
                                   crop_unit='pixel')
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [1, 9], atol=0.1)
        assert np.allclose(ax.get_ylim(), [1, 13], atol=0.1)

        hdul.close()

    def test_cd_wcs(self, capsys, cd_wcs_fits_data):
        hdul = cd_wcs_fits_data

        fig = quicklook.make_image(hdul, crop_region=None)
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [0, 9], atol=0.5)
        assert np.allclose(ax.get_ylim(), [0, 9], atol=0.5)

        assert 'No celestial WCS' not in capsys.readouterr().err
        assert ax.get_xlabel() == 'RA (J2000)'
        assert ax.get_ylabel() == 'Dec (J2000)'

        fig = quicklook.make_image(hdul, crop_region=[5, 7, 4, 6])
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [-2.4, 0.4], atol=0.1)
        assert np.allclose(ax.get_ylim(), [3.9, 8.2], atol=0.1)

        hdul.close()

    def test_bad_wcs(self, capsys, wcs_fits_data):
        hdul = wcs_fits_data
        # trigger error in WCS object
        hdul[0].header['CRVAL2'] = -9999
        fig = quicklook.make_image(hdul)
        assert 'Unreadable WCS' in capsys.readouterr().err
        assert fig.get_axes()[0].get_xlabel() == 'x'
        assert fig.get_axes()[0].get_ylabel() == 'y'
        hdul.close()

    def test_grid(self, simple_fits_data, cd_wcs_fits_data, figures_same):
        hdul = simple_fits_data
        # there's no simple test for grid on/off in matplotlib,
        # so just test that the figure is different with and without it
        fig1 = quicklook.make_image(hdul, grid=False)
        fig2 = quicklook.make_image(hdul, grid=True)
        assert not figures_same(fig1, fig2)
        hdul.close()

        # for rotated data, grid is always on
        hdul = cd_wcs_fits_data
        fig1 = quicklook.make_image(hdul, grid=False)
        fig2 = quicklook.make_image(hdul, grid=True)
        assert figures_same(fig1, fig2)
        hdul.close()

    def test_contours(self, capsys, simple_fits_data, figures_same):
        hdul = simple_fits_data
        fig1 = quicklook.make_image(hdul, n_contour=0)
        assert 'Contours' not in capsys.readouterr().out

        fig2 = quicklook.make_image(hdul, n_contour=3, fill_contours=False)
        assert 'Contours: [ 0.2475 49.5    98.7525]' in capsys.readouterr().out
        assert not figures_same(fig1, fig2)

        fig3 = quicklook.make_image(hdul, n_contour=3, fill_contours=True)
        assert 'Contours: [ 0.2475 49.5    98.7525]' in capsys.readouterr().out
        assert not figures_same(fig2, fig3)
        hdul.close()

    def test_beam(self, capsys, wcs_fits_data, figures_same):
        # test beam marker
        hdul = wcs_fits_data
        fig1 = quicklook.make_image(hdul, beam=False)
        assert 'Beam' not in capsys.readouterr().out

        fig2 = quicklook.make_image(hdul, beam=True)
        assert 'Beam' in capsys.readouterr().out
        assert not figures_same(fig1, fig2)

        # remove beam keywords - should issue warning only
        del hdul[0].header['BMAJ']
        fig3 = quicklook.make_image(hdul, beam=True)
        assert 'Beam keywords not found' in capsys.readouterr().err
        assert figures_same(fig1, fig3)
        hdul.close()

    def test_decimal(self, capsys, wcs_fits_data, figures_same):
        # test option to display units in decimal degrees
        hdul = wcs_fits_data
        fig1 = quicklook.make_image(hdul, decimal=False)
        fig2 = quicklook.make_image(hdul, decimal=True)
        assert not figures_same(fig1, fig2)

        assert fig1.axes[0].coords[0].get_format_unit() == 'hourangle'
        assert fig2.axes[0].coords[0].get_format_unit() == 'degree'
        hdul.close()

    def test_spectral_cube(self, capsys, cube_fits_data):
        # test making image from a slice of cube data
        hdul = cube_fits_data

        # fails without slice keyword
        with pytest.raises(ValueError) as err:
            quicklook.make_image(hdul)
        assert 'Slice must be set' in str(err)

        fig = quicklook.make_image(hdul, cube_slice=5, crop_region=None)

        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [0, 9], atol=0.5)
        assert np.allclose(ax.get_ylim(), [0, 9], atol=0.5)

        # non celestial WCS, with units
        assert ax.get_xlabel() == 'x (arcsec)'
        assert ax.get_ylabel() == 'y (arcsec)'

        fig = quicklook.make_image(hdul, cube_slice=5,
                                   crop_region=[5, 7, 4, 6])
        ax = fig.get_axes()[0]
        assert np.allclose(ax.get_xlim(), [3, 7], atol=0.1)
        assert np.allclose(ax.get_ylim(), [4, 10], atol=0.1)
        hdul.close()

    def test_plot_layout(self, simple_fits_data):
        # test option to allow subplots to be added to the figure - the
        # image is always the first subplot
        hdul = simple_fits_data
        fig1 = quicklook.make_image(hdul, plot_layout=None)
        assert fig1.axes[0].get_geometry() == (1, 1, 1)
        fig2 = quicklook.make_image(hdul, plot_layout=(2, 1))
        assert fig2.axes[0].get_geometry() == (2, 1, 1)
        fig3 = quicklook.make_image(hdul, plot_layout=(1, 2))
        assert fig3.axes[0].get_geometry() == (1, 2, 1)
        hdul.close()

    def test_watermark(self, simple_fits_data):
        hdul = simple_fits_data

        fig = quicklook.make_image(hdul, watermark='')
        ax = fig.get_axes()[0]
        assert len(ax.texts) == 0

        fig = quicklook.make_image(hdul, watermark='watermark')
        ax = fig.get_axes()[0]
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == 'watermark'
        hdul.close()


class TestQuicklookPlot(object):

    def test_one_plot(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1])
        assert len(ax.lines) == 1
        assert ax.legend_ is None
        hdul.close()

    def test_multi_plot(self, spectrum_fits_data):
        hdul = spectrum_fits_data

        # no labels: argument is passed directly to ax.step
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1:4].T,
                                     labels=None)
        assert len(fig.get_axes()) == 1
        assert len(ax.lines) == 3
        assert ax.legend_ is None

        # with labels: spectra are iterated over, must match wavelengths,
        # data, errors, and labels
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        quicklook.make_spectral_plot(ax, [hdul[0].data[0]] * 3,
                                     hdul[0].data[1:4],
                                     spectral_error=hdul[0].data[1:4],
                                     labels=['1', '2', '3'])
        assert len(fig.get_axes()) == 1
        assert len(ax.lines) == 3
        assert len(ax.collections) == 3
        assert ax.legend_ is not None
        assert len(ax.legend_.get_lines()) == 3

        # with labels and many spectra
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        quicklook.make_spectral_plot(ax, [hdul[0].data[0]] * 20,
                                     np.array([hdul[0].data[1]] * 20),
                                     labels=list(np.arange(20)))
        assert len(fig.get_axes()) == 1
        assert len(ax.lines) == 20
        assert ax.legend_ is not None
        # legend strings are capped at 15 + 2
        assert len(ax.legend_.get_lines()) == 17
        hdul.close()

    def test_overplot(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     overplot=hdul[0].data[2],
                                     overplot_label='test')
        axes = fig.get_axes()
        assert len(axes) == 2
        assert axes[0] == ax
        assert len(ax.lines) == 1
        assert len(axes[1].lines) == 1
        assert len(ax.collections) == 0
        hdul.close()

    def test_error_spec(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     spectral_error=hdul[0].data[2])
        assert len(fig.get_axes()) == 1
        assert len(ax.lines) == 1
        assert len(ax.collections) == 1
        hdul.close()

    def test_scale(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        # no scale: takes full range with some padding
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     scale=None)
        ylim = ax.get_ylim()
        assert ylim[0] < np.min(hdul[0].data[1])
        assert ylim[1] > np.max(hdul[0].data[1])

        # scale with percentile limits
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     scale=[0, 100])
        ylim = ax.get_ylim()
        assert ylim[0] == np.min(hdul[0].data[1])
        assert ylim[1] == np.max(hdul[0].data[1])

        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     scale=[20, 80])
        ylim = ax.get_ylim()
        assert ylim[0] > np.min(hdul[0].data[1])
        assert ylim[1] < np.max(hdul[0].data[1])
        hdul.close()

    def test_marker(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        # markers are passed as [[x], [y]] lists
        marker = [[2, 4, 6], [102, 104, 106]]
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     marker=marker)
        assert len(ax.collections) == 1
        assert isinstance(ax.collections[0], mpl.collections.PathCollection)
        hdul.close()

    def test_labels(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        # no unit string
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     xunit='', yunit='')
        assert ax.get_xlabel() == 'Wavelength'
        assert ax.get_ylabel() == 'Spectral flux'

        # wavenumber (cm-1 only)
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     xunit='cm-1', yunit='normalized')
        assert ax.get_xlabel() == 'Wavenumber (cm-1)'
        assert ax.get_ylabel() == 'Spectral flux (normalized)'

        # wavelength
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     xunit='um', yunit='Jy')
        assert ax.get_xlabel() == 'Wavelength (um)'
        assert ax.get_ylabel() == 'Spectral flux (Jy)'
        hdul.close()

    def test_title(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        # no title
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     title='')
        assert ax.title.get_text() == ''

        # with title
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     title='Test title')
        assert ax.title.get_text() == 'Test title'
        hdul.close()

    def test_watermark(self, spectrum_fits_data):
        hdul = spectrum_fits_data
        fig = mpl.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)

        # no watermark
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     watermark='')
        assert len(ax.texts) == 0

        # with watermark
        quicklook.make_spectral_plot(ax, hdul[0].data[0], hdul[0].data[1],
                                     watermark='watermark')
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == 'watermark'
        hdul.close()

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.steppoldip import StepPolDip
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestPolDip(DRPTestCase):
    def test_poldip(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)

        step = StepPolDip()
        out = step(prep)
        assert isinstance(out, DataFits)

        # check for the important planes
        assert 'Q (INSTRUMENTAL)' in out.imgnames
        assert 'U (INSTRUMENTAL)' in out.imgnames

        # check for assc* keys
        assert 'ASSC_AOR' in out.header
        assert 'ASSC_MSN' in out.header

        # run again without aor/missn
        prep.delheadval('AOR_ID')
        prep.delheadval('MISSN-ID')
        out = step(prep)
        assert 'ASSC_AOR' not in out.header
        assert 'ASSC_MSN' not in out.header

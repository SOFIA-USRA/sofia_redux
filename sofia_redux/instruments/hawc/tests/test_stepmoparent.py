# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase


class TestStepMOParent(DRPTestCase):
    def test_miso(self):
        step = StepMOParent()
        assert step.iomode == 'MIMO'
        assert step.procname == 'unk'

        # default run returns datain in dataout
        step.datain = [1, 2, 3]
        step.run()
        assert step.dataout == [1, 2, 3]

    def test_runend(self):
        # test that run end handles list of output
        df1 = DataFits()
        df2 = DataFits()
        step = StepMOParent()
        step.runend([df1, df2])
        assert df1.header['PRODTYPE'] == 'parentmo'
        assert df2.header['PRODTYPE'] == 'parentmo'

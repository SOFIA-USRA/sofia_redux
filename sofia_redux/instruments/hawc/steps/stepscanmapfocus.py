# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Focus image reconstruction pipeline step."""

from astropy import log

from sofia_redux.instruments.hawc.dataparent import DataParent
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap

__all__ = ['StepScanMapFocus']


class StepScanMapFocus(StepMOParent):
    """
    Reconstruct an image from short focus scans.

    This step calls the scan map data reduction package as an external
    process to reduce focus scans data.

    Data are grouped by the focus value before being passed to scan map.
    Each set of focus values produces a single scan map output file with
    image extensions SIGNAL, EXPOSURE, NOISE and S/N.

    Scan map parameters are not generally modified for this step. They
    are used as defined in the configuration files for mode_focus
    data.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanmapfocus', and are named with
        the step abbreviation 'SMP'.

        Parameters defined for this step are:

        groupkeys : str
            List of header keywords to decide data group
            membership (| separated).
        groupkfmt : str
            List of group key formats for string
            comparison (unused if "", | separated).
        """
        # Name of the pipeline reduction step
        self.name = 'scanmapfocus'
        self.description = 'Construct Scan Map'

        # procname is taken from the redstep below
        self.procname = 'smp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['groupkeys', 'FOCUS_ST',
                               'List of header keywords to decide data '
                               'group membership (| separated)'])
        self.paramlist.append(['groupkfmt', '%.1f',
                               'List of group key formats for string '
                               'comparison (unused if "", | separated)'])

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multi-in multi-out (MIMO) step:
        self.datain should be a list of DataFits, and output
        will be a single DataFits, stored in self.dataout.

        The process is:

        1. Group input data by the header keyword FOCUS_ST.
        2. Call StepScanMap for each input group.

        """

        # Get the step to run on the group
        redstep = StepScanMap()

        # Set up data groups, get keys and key formats
        datagroups = []
        groupkeys = self.getarg('groupkeys').split('|')
        groupkfmt = self.getarg('groupkfmt')
        if len(groupkfmt) == 0:
            groupkfmt = None
        else:
            groupkfmt = groupkfmt.split('|')

        # Loop over files
        for data in self.datain:
            groupind = 0
            # Loop over groups until group match found or end reached
            while groupind < len(datagroups):
                # Check if data fits group
                found = True
                gdata = datagroups[groupind][0]
                for keyi in range(len(groupkeys)):
                    # Get key from group and new data - format if needed
                    key = groupkeys[keyi]
                    dkey = data.getheadval(key)
                    gkey = gdata.getheadval(key)
                    if groupkfmt is not None:
                        dkey = groupkfmt[keyi] % dkey
                        gkey = groupkfmt[keyi] % gkey

                    # Compare
                    if dkey != gkey:
                        found = False

                # Found -> add to group
                if found:
                    datagroups[groupind].append(data)
                    break

                # Not found -> increase group index
                groupind += 1

            # If not in any group -> make new group
            if groupind == len(datagroups):
                datagroups.append([data, ])

        # info messages
        log.debug(" Found %d data groups" % len(datagroups))
        for groupind in range(len(datagroups)):
            group = datagroups[groupind]
            msg = "  Group %d len=%d" % (groupind, len(group))
            for key in groupkeys:
                msg += " %s = %s" % (key, group[0].getheadval(key))
            log.debug(msg)

        # Reduce input files - collect output files
        self.dataout = []

        # Loop over groups -> save output in self.dataout
        for groupi in range(len(datagroups)):
            group = datagroups[groupi]

            # Reduce the data
            dataout = redstep(group)

            # add output to dataout
            if issubclass(dataout.__class__, DataParent):
                self.dataout.append(dataout)
            else:
                for data in dataout:
                    self.dataout.append(data)

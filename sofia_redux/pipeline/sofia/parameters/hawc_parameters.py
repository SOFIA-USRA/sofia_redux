# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC parameter sets."""

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.pipeline.parameters import Parameters


# these step lists override the config in the hawc pipeline config,
# in order to call special redux-defined super steps
STEPLISTS = {
    # primary nod and nodpol modes: make flats if necessary,
    # then prepare/demodulate, then run the rest of the reduction
    'nodpol': ['make_flats', 'demodulate',
               'StepDmdPlot', 'StepDmdCut', 'StepFlat',
               'StepShift', 'StepSplit', 'StepCombine',
               'StepNodPolSub', 'StepStokes', 'StepWcs',
               'StepIP', 'StepRotate', 'StepOpacity',
               'StepCalibrate', 'StepBgSubtract', 'StepBinPixels',
               'StepMerge', 'StepPolVec', 'StepRegion', 'StepPolMap'],
    'nod': ['make_flats', 'demodulate',
            'StepDmdPlot', 'StepDmdCut', 'StepFlat',
            'StepShift', 'StepSplit', 'StepCombine',
            'StepNodPolSub', 'StepStokes', 'StepWcs',
            'StepOpacity', 'StepBgSubtract', 'StepBinPixels',
            'StepMerge', 'StepCalibrate', 'StepImgMap'],
    # intermediate nod and nodpol modes, starting from Level 1
    # demodulated data: still make flats if necessary,
    # then continue reduction
    'nodpol_dmd': ['make_flats',
                   'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                   'StepShift', 'StepSplit', 'StepCombine',
                   'StepNodPolSub', 'StepStokes', 'StepWcs',
                   'StepIP', 'StepRotate', 'StepOpacity',
                   'StepCalibrate', 'StepBgSubtract', 'StepBinPixels',
                   'StepMerge', 'StepPolVec', 'StepRegion',
                   'StepPolMap'],
    'nod_dmd': ['make_flats',
                'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                'StepShift', 'StepSplit', 'StepCombine',
                'StepNodPolSub', 'StepStokes', 'StepWcs',
                'StepOpacity', 'StepBgSubtract', 'StepBinPixels',
                'StepMerge', 'StepCalibrate', 'StepImgMap'],
    # now the same, but for flux standards -- they have an extra
    # photometry/calibration step instead of regular calibration
    'nodpol_std': ['make_flats', 'demodulate',
                   'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                   'StepShift', 'StepSplit', 'StepCombine',
                   'StepNodPolSub', 'StepStokes', 'StepWcs',
                   'StepIP', 'StepRotate', 'StepOpacity',
                   'StepBgSubtract', 'StepMerge',
                   'StepStdPhotCal', 'StepPolVec',
                   'StepRegion', 'StepPolMap'],
    'nod_std': ['make_flats', 'demodulate',
                'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                'StepShift', 'StepSplit', 'StepCombine',
                'StepNodPolSub', 'StepStokes', 'StepWcs',
                'StepOpacity', 'StepBgSubtract',
                'StepMerge', 'StepStdPhotCal', 'StepImgMap'],
    'nodpol_std_dmd': ['make_flats',
                       'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                       'StepShift', 'StepSplit', 'StepCombine',
                       'StepNodPolSub', 'StepStokes', 'StepWcs',
                       'StepIP', 'StepRotate', 'StepOpacity',
                       'StepBgSubtract', 'StepMerge',
                       'StepStdPhotCal', 'StepPolVec', 'StepRegion',
                       'StepPolMap'],
    'nod_std_dmd': ['make_flats',
                    'StepDmdPlot', 'StepDmdCut', 'StepFlat',
                    'StepShift', 'StepSplit', 'StepCombine',
                    'StepNodPolSub', 'StepStokes', 'StepWcs',
                    'StepOpacity', 'StepBgSubtract',
                    'StepMerge', 'StepStdPhotCal', 'StepImgMap'],
    # additional calibration mode for skycals: process intcal,
    # then make skycal
    'skycal': ['process_intcal', 'StepCheckhead', 'StepScanMapFlat',
               'StepSkycal'],
}


class HAWCParameters(Parameters):
    """Reduction parameters for the HAWC pipeline."""
    def __init__(self, config=None, param_lists=None, mode=None):
        """
        Initialize parameters from HAWC DRP step defaults.

        Parameters
        ----------
        config : `configobj.ConfigObj` or dict-like, optional
            DRP configuration object.
        param_lists : dict
            DRP pipeline step parameter lists for all available steps.
            Keys are pipeline step names.
        """
        # read default pipeline parameters from parameter lists
        # extracted from the pipeline steps
        default = self.read_parameters(param_lists)

        # update them from the config file
        default = self.update_parameters(default, config=config)

        # initialize with the default dictionary
        super().__init__(default)

        # track an override mode, if necessary
        self.override_mode = mode

    def read_parameters(self, param_lists):
        """
        Read parameters from DRP parameter lists.

        DRP parameter types are inferred from their default values
        (see `Parameters.get_param_type`).

        A standard set of Redux control parameters are also added to each
        pipeline step (save, undo, display, load, keep_auxout).  These
        may be later overridden by methods in this class.

        Parameters
        ----------
        param_lists : dict
            DRP pipeline step parameter lists for all available steps.
            Keys are pipeline step names.

        Returns
        -------
        default : dict
            Mapping of step names to parameter set values.
            Keys are pipeline step names; values are dicts with
            `ParameterSet` keywords.
        """
        default = {}
        if param_lists is None:
            return default

        for step_name, par_list in param_lists.items():
            default[step_name] = []

            # always add save/undo/display parameters
            for pset in self.standard_param():
                default[step_name].append(pset)

            for par in par_list:
                key, value, description = par
                dtype = self.get_param_type(value)
                if dtype == 'bool':
                    wtype = 'check_box'
                else:
                    wtype = 'text_box'
                pdict = {
                    'key': key,
                    'value': self.fix_param_type(value, dtype),
                    'description': description,
                    'dtype': dtype,
                    'wtype': wtype
                }
                default[step_name].append(pdict)
        return default

    def standard_param(self):
        """
        Return a standard set of parameters to add to all steps.

        The default is:

        * don't save output from the step
        * allow undo of the step
        * display the output of the step
        * load the input data from memory if it has not
          already been loaded
        * don't keep auxiliary output from the previous step
          for re-display after this step

        """
        save = {
            'key': 'save',
            'value': False,
            'description': 'Save output data',
            'dtype': 'bool',
            'wtype': 'check_box'
        }
        undo = {
            'key': 'undo',
            'value': True,
            'description': 'Allow undo',
            'dtype': 'bool',
            'wtype': 'check_box',
            'hidden': True
        }
        display = {
            'key': 'display',
            'value': True,
            'description': 'Display output data',
            'dtype': 'bool',
            'wtype': 'check_box',
            'hidden': True
        }
        load = {
            'key': 'load',
            'value': True,
            'description': 'Load input data into memory',
            'dtype': 'bool',
            'wtype': 'check_box',
            'hidden': True
        }
        keep_auxout = {
            'key': 'keep_auxout',
            'value': False,
            'description': 'Keep auxout data for next step',
            'dtype': 'bool',
            'wtype': 'check_box',
            'hidden': True
        }
        return [save, undo, display, load, keep_auxout]

    def update_parameters(self, default, config=None):
        """
        Update parameter values from a DRP configuration.

        Parameters
        ----------
        default : dict
            Default parameter dictionary to update.  Updates are
            made in place.
        config : `configobj.ConfigObj`, optional
            DRP configuration object.  If not provided, the default
            configuration will be used.

        Returns
        -------
        dict
            Updated default parameter dictionary.
        """
        df = DataFits(config=config)
        for key, pset in df.config.items():
            if key not in default:
                continue
            for pkey in pset:
                for init_par in default[key]:
                    if init_par['key'].lower() == pkey.lower():
                        pval = pset[pkey]
                        init_par['value'] = \
                            self.fix_param_type(pval, init_par['dtype'])
                        break

        return default

    def to_config(self):
        """
        Read parameter values into a configuration object.

        Section names in the output object are written as
        ``stepindex: stepname`` in order to record the order of
        reduction steps, and to keep any repeated step names uniquely
        identified.  Only the current parameter values are recorded.
        Other information, such as data or widget type or default
        values, is lost.

        Overrides parent function in order to add an override mode
        flag to the top-level configuration.

        Returns
        -------
        ConfigObj
            The parameter values in a `configobj.ConfigObj` object.
        """
        config = super().to_config()
        # add mode if necessary
        if self.override_mode is not None:
            config['mode'] = self.override_mode
        return config

    # modify defaults for some steps

    def checkhead(self, step_index):
        """
        Modify parameters for the checkhead step.

        This step only checks headers.  Don't save, load, or
        display the data.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("load", False)
        self.current[step_index].set_value("save", False)
        self.current[step_index]["save"]["hidden"] = True

    def fluxjump(self, step_index):
        """
        Modify parameters for the fluxjump step.

        No undo or display for raw data.  Do load, don't save.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("load", True)
        self.current[step_index].set_value("save", False)

    def prepare(self, step_index):
        """
        Modify parameters for the prepare step.

        No undo or display for raw data.  Do load, don't save.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("load", True)
        self.current[step_index].set_value("save", False)

    def dmdcut(self, step_index):
        """
        Modify parameters for the dmdcut step.

        No display for demodulated data.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("display", False)

    def dmdplot(self, step_index):
        """
        Modify parameters for the dmdplot step.

        No display for demodulated data.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("display", False)

    def opacity(self, step_index):
        """
        Modify parameters for the opacity step.

        Save the output data by default.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("save", True)

    def merge(self, step_index):
        """
        Modify parameters for the merge step.

        Save the output data by default.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("save", True)

    def stdphotcal(self, step_index):
        """
        Modify parameters for the stdphotcal step.

        Save the output data by default.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("save", True)

    def calibrate(self, step_index):
        """
        Modify parameters for the calibrate step.

        Save the output data by default.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("save", True)

    def polmap(self, step_index):
        """
        Modify parameters for the polmap step.

        Save the output data by default.  Keep the auxout
        from the previous step (DS9 regions).

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("keep_auxout", True)

    def scanmap(self, step_index):
        """
        Modify parameters for the scanmap step.

        Don't load data into memory: scanmap loads it separately.
        If no output is expected, don't save or display the output.
        If output is expected, do save and display it.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # never load data -- it will do it itself.
        self.current[step_index].set_value("load", False)

        try:
            no_out = self.current[step_index].get_value("noout")
        except KeyError:
            no_out = False

        if no_out:
            # no output is expected:
            # set save to False and hide it -- attempting to
            # save this data will crash the pipeline.
            self.current[step_index].set_value("save", False)
            self.current[step_index]["save"]["hidden"] = True
            # also don't try to display
            self.current[step_index].set_value("display", False)
        else:
            self.current[step_index].set_value("save", True)
            self.current[step_index].set_value("display", True)

    def scanmappol(self, step_index):
        """
        Modify parameters for the scanmappol step.

        Don't load data into memory: scanmap loads it separately.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # never load data -- it will do it itself.
        self.current[step_index].set_value("load", False)

        # save and display data
        self.current[step_index].set_value("save", True)
        self.current[step_index].set_value("display", True)

    def scanmapflat(self, step_index):
        """
        Modify parameters for the scanmapflat step.

        Don't load data into memory: scanmap loads it separately.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # never load data for scanmap -- it will do it itself.
        self.current[step_index].set_value("load", False)

        # save data
        self.current[step_index].set_value("save", True)

    def skycal(self, step_index):
        """
        Modify parameters for the skycal step.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # save data
        self.current[step_index].set_value("save", True)

    def skydip(self, step_index):
        """
        Modify parameters for the skydip step.

        No output FITS file; hide save, don't display.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("save", False)
        self.current[step_index]["save"]["hidden"] = True

    def poldip(self, step_index):
        """
        Modify parameters for the poldip step.

        Save the output by default.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("save", True)

    def scanmapfocus(self, step_index):
        """
        Modify parameters for the scanmapfocus step.

        This step calls scanmap: don't load the input data.
        Do save the output.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("load", False)
        self.current[step_index].set_value("save", True)

    def focus(self, step_index):
        """
        Modify parameters for the focus step.

        No output FITS file; hide save, don't display.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("save", False)
        self.current[step_index]["save"]["hidden"] = True

    # add param definitions for some custom steps

    def make_flats(self, step_index):
        """
        Define parameters for the make_flats super-step.

        Parameters for the mkflat step are added.  Earlier steps
        (prepare, demodulate, etc.) are not, since these parameters
        generally do not need modification by the user.  They
        can still be overridden by modifying a DRP configuration
        file.

        Save is off by default for the final DCL file.  Don't allow
        undo or display; the auxiliary OFT file will still display.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # add the standard parameters
        for pset in self.standard_param():
            self.current[step_index].set_param(**pset)

        # add the mkflat parameters; the rest shouldn't be needed
        if 'mkflat' in self.default:
            for pkey, pset in self.default['mkflat'].items():
                if pkey in ['save', 'undo', 'display']:
                    continue
                self.current[step_index].set_param(key=pkey, **pset)

        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", False)
        self.current[step_index].set_value("keep_auxout", True)

        # for this step, "save" means save the DCL file only
        self.current[step_index]["save"]["value"] = False
        self.current[step_index]["save"]["name"] = "Save DCL file"
        self.current[step_index]["save"]["description"] = \
            "Save final output file from flat process\n" \
            "(may be useful for instrument diagnosis).\n" \
            "OFT file is always saved."

    def process_intcal(self, step_index):
        """
        Define parameters for the process_intcal super-step.

        Parameters for the mkflat step are added.  Earlier steps
        (prepare, demodulate, etc.) are not, since these parameters
        generally do not need modification by the user.  They
        can still be overridden by modifying a DRP configuration
        file.

        Save is on by default for the final DCL file.  Don't allow undo.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # add the standard parameters
        for pset in self.standard_param():
            self.current[step_index].set_param(**pset)

        # add the mkflat parameters; the rest shouldn't be needed
        if 'mkflat' in self.default:
            for pkey, pset in self.default['mkflat'].items():
                if pkey in ['save', 'undo', 'display']:
                    continue
                elif pkey == 'dcl_only':
                    pset['value'] = True
                elif pkey == 'flatoutfolder':
                    pset['value'] = 'intcals'
                self.current[step_index].set_param(key=pkey, **pset)

        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", True)
        self.current[step_index].set_value("keep_auxout", True)

        # for this step, "save" means save the DCL file only,
        # and it should be saved to the flats folder instead
        self.current[step_index]["save"]["value"] = False
        self.current[step_index]["save"]["hidden"] = True
        self.current[step_index]["save"]["name"] = "Save DCL file"

    def demodulate(self, step_index):
        """
        Define parameters for the demodulate super-step.

        Start with the parameters for the DRP demodulate step,
        then add in the abort parameter for the checkhead step.
        The final DMD file is saved by default; undo and display
        are turned off.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        # add the header check abort parameter
        if 'checkhead' in self.default:
            pset = self.default['checkhead']['abort']
            self.current[step_index].set_param(key='abort', **pset)
        else:
            self.current[step_index].set_param(key='abort', value=True)
        self.current[step_index]["abort"]["name"] = "Abort for bad headers"

        # set undo, display, and load to false
        self.current[step_index].set_value("undo", False)
        self.current[step_index].set_value("display", False)

        # for this super step, "save" means save the DMD file only.
        # Earlier sub-steps should never be saved.
        self.current[step_index].set_value("save", True)
        self.current[step_index]["save"]["name"] = "Save DMD file"

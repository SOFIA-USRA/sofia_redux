# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC Reduction pipeline steps"""

import gc
import importlib
import os
import warnings
from copy import deepcopy

from astropy import log

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.gui.qad_viewer import QADViewer
from sofia_redux.pipeline.sofia.parameters.hawc_parameters \
    import HAWCParameters, STEPLISTS

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    from sofia_redux.instruments import hawc
except ImportError:
    raise SOFIAImportError('HAWC modules not installed')

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc import steps

# this import is not used here but is needed to avoid
# a numba bug on Linux systems
from sofia_redux.instruments.hawc.steps import basehawc
assert basehawc


class HAWCReduction(Reduction):
    """
    HAWC+ reduction steps.

    Reduction algorithms are all defined in the
    `sofia_redux.instruments.hawc` module.

    This reduction object does not define a method for each pipeline
    step.  Rather, it calls run_drp_step each time step is called,
    and uses the DRP infrastructure to determine which step to run.

    Some reduction "steps" are defined in this class as super-steps,
    combining several DRP steps together.  This may be used for convenience
    (e.g. the `make_flats` step, which saves the user from running a
    separate pipeline to produce flat files in nod or nod-pol modes) or
    for performance (e.g. the `demodulate` step, which loads one large
    raw file at a time, so that all data is not in memory at once).

    Attributes
    ----------
    step_class : dict
        All step classes available from the DRP. Keys are
        pipeline step names.
    step_name : dict
        Names of all pipeline steps available from the DRP. Keys
        are pipeline step class names.
    param_lists : dict
        DRP pipeline step parameter lists for all available steps.
        Keys are pipeline step names.
    extra_steps : dict
        Locally defined pipeline steps.  Keys are step names,
        values are step descriptions.
    override_steplist : dict
        Pipeline step lists to override the ones defined in the
        DRP configuration files.  These lists should include any
        locally defined steps that replace DRP steps.
        Keys are pipeline modes, values are lists of steps to
        run.
    intermediate : bool
        If True, the input data is an intermediate product, and
        a subset of the full pipeline recipe is being run.
    auxout : list of str
        Auxiliary output files produced by the pipeline, intended
        for display.  These may include PNG images and DS9 region
        files.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes
        self.name = "DRP"
        self.instrument = "HAWC"
        self.mode = "any"
        self.data_keys = ['File Name', 'OBJECT',
                          'INSTCFG', 'INSTMODE', 'CALMODE',
                          'OBSTYPE', 'SPECTEL1', 'SPECTEL2',
                          'ALTI_STA', 'ZA_START', 'EXPTIME',
                          'PRODTYPE', 'FILEGPID', 'SCRIPTID']

        self.pipe_name = "HAWC_DRP"
        self.pipe_version = hawc.__version__.replace('.', '_')

        # recipe and parameters will be defined later,
        # when files are loaded
        self.parameters = None
        self.recipe = []

        # set an environment variable used by the pipeline
        # to determine relative paths to reference files
        os.environ['DPS_HAWCPIPE'] = os.path.dirname(hawc.__file__)

        # load steps from the hawc.steps module
        self.step_class, self.step_name, \
            self.processing_steps, self.param_lists = self.load_step_packs()

        # describe a few extra locally-defined steps
        self.extra_steps = {'make_flats': 'Make Flats',
                            'demodulate': 'Demodulate Chops',
                            'process_intcal': 'Process INTCAL'}
        self.override_steplist = STEPLISTS

        # don't keep a data copy in memory for initial steps
        self.allow_undo = False

        # raise an error if input list is empty
        self.check_input = True

        # reduction variables
        self.override_mode = None
        self.intermediate = False
        self.output_directory = os.getcwd()
        self.auxout = []

        # ignore warnings from pipeline steps (numpy, scipy, astropy, etc.)
        warnings.filterwarnings('ignore')

    def load(self, data):
        """
        Load input data to make it available to reduction steps.

        Data files are loaded into new DRP DataFits objects.  From
        these objects, the pipeline mode and steps are determined.
        The configuration file in the DRP package (plus any relevant
        date-specific or user overrides) is used to determine the
        default parameters for all pipeline steps.

        After loading, headers are read from all input data and
        stored in the display_data attribute for display in the QAD
        header viewer.  The raw data is not displayed.

        Parameters
        ----------
        data : `list` of str
            Input data file names to be loaded.
        """
        # call the parent method
        super().load(data)

        self.input = []
        self.mode = None
        msgs = []
        for i, datafile in enumerate(data):
            df = DataFits()
            df.loadhead(datafile, dataname='all')
            if df.mode is None and not self.override_mode:
                raise ValueError("Pipeline mode not found "
                                 "for {}".format(datafile))

            # special case: mode=intcal is used to make flats,
            # but is not the primary mode, unless only intcals are
            # loaded
            if self.mode is None and \
                    (df.mode != 'intcal' or i == len(data) - 1):

                # override primary mode if necessary
                if self.override_mode:
                    override_mode = str(self.override_mode)
                    mode_str = f'mode_{override_mode}'
                    if mode_str in df.config \
                            or mode_str in self.override_steplist:
                        df.mode = override_mode
                        df.mergeconfig(mode=override_mode)
                    else:
                        raise ValueError(f"Pipeline override mode "
                                         f"{override_mode} not found")

                # use the primary mode to define the recipe and parameters
                msgs.append("Pipeline mode: {}".format(df.mode))
                self.mode = df.mode
                if self.mode in self.override_steplist:
                    stepnames = self.override_steplist[self.mode]
                else:
                    stepnames = \
                        df.config['mode_{}'.format(df.mode)]['stepslist']

                self.recipe = []
                for j, step in enumerate(stepnames):
                    # if step is locally defined, use it
                    if hasattr(self, step):
                        self.recipe.append(step)
                        try:
                            self.processing_steps[step] = \
                                self.extra_steps[step]
                        except KeyError:
                            self.processing_steps[step] = step
                    elif step in self.step_name:
                        # otherwise, use the version in the DRP steps
                        self.recipe.append(self.step_name[step])
                    else:
                        # if not found, raise an error
                        raise ValueError("Pipeline step {} "
                                         "not found".format(step))

                # check for intermediate file: modify recipe if necessary
                try:
                    prodtype = df.getheadval('PRODTYPE', errmsg=False)
                except KeyError:
                    prodtype = 'unknown'

                # check for intermediate step in recipe, with
                # an exception for re-running the final image
                prodtype = str(prodtype).strip().lower()
                if prodtype in self.recipe \
                        and prodtype not in ['polmap', 'imgmap']:
                    idx = self.recipe.index(prodtype)

                    # recipe is all steps after the input one
                    self.recipe = self.recipe[idx + 1:]

                    if len(self.recipe) == 0:
                        raise ValueError("No steps to run for "
                                         "prodtype {}.".format(prodtype))

                    # set a flag to mark the recipe as modified
                    self.intermediate = True

                # compose a message for the steps and the config files used,
                # for logging after all files have loaded
                msgs.append("Processing steps: {}".format(self.recipe))
                msgs.append("Config files:")
                for cfile in df.config_files:
                    msgs.append("    {}".format(cfile))

                # update redux parameters from new config
                self.parameters = HAWCParameters(config=df.config,
                                                 param_lists=self.param_lists,
                                                 mode=self.override_mode)
                self.load_parameters()

            log.info("Input: {}".format(datafile))
            self.input.append(df)

        # log the mode and config files
        for msg in msgs:
            log.info(msg)

        # pass headers to viewer
        self.auxout = []
        self.set_display_data()

    def load_step_packs(self):
        """
        Load DRP pipeline step modules.

        Returns
        -------
        step_class : dict
            All step classes available from the DRP. Keys are
            pipeline step names.
        step_name : dict
            Names of all pipeline steps available from the DRP. Keys
            are pipeline step class names.
        processing_steps : dict
            Display names for pipeline steps.  Keys are pipeline step
            names.
        param_lists : dict
            DRP pipeline step parameter lists for all available steps.
            Keys are pipeline step names.
        """
        step_class = {}
        step_name = {}
        step_desc = {}
        param_lists = {}
        for class_name in steps.__all__:
            cls = getattr(steps, class_name)
            if issubclass(cls, StepParent) and \
                    'parent' not in class_name.lower():
                step = cls()
                class_name = cls.__name__
                step_class[step.name] = class_name
                step_name[class_name] = step.name
                step_desc[step.name] = step.description
                param_lists[step.name] = step.paramlist

        return step_class, step_name, step_desc, param_lists

    def register_viewers(self):
        """
        Instantiate viewers appropriate to the reduction.

        This method instantiates and returns a `QADViewer` object, used
        to display data from this reduction in DS9.  Data for the viewer
        is stored in the `display_data` attribute.

        Returns
        -------
        `list` of `Viewer`
            All viewers supported by this reduction object.
        """
        viewers = [QADViewer()]
        return viewers

    def run_drp_step(self, step_name=None, use_param=True):
        """
        Run a DRP pipeline step.

        Pipeline steps are assumed to be implemented in a module of
        the name 'step' + `step_name`, in the
        `sofia_redux.instruments.hawc.steps` package.
        The step class name is stored in the `step_class` attribute.

        The step class is imported, instantiated, then called on
        the data in the input attribute.  If the data have not yet
        been loaded from disk, they are loaded at this time.  If the
        pipeline step is a single-input step, it is called in a loop,
        once per input file.  If it is a multi-input step, it is called
        once on all the input data.

        Whether the output data is saved to disk or displayed after the step
        completes is controlled by a Redux parameter, applied to each
        step via the `HAWCParameters` class.

        At the end of this method, output data are stored in the input
        attribute, so that they are available for the next processing
        step.

        Parameters
        ----------
        step_name : str, optional
            Name of the step to run.  If not provided, the current
            step_index will be used to retrieve the step from the
            recipe.
        use_param : bool, optional
            If False, parameters from the Redux GUI will not
            be passed to the pipeline step.  This is used for some
            sub-steps in the locally defined reduction steps.

        Raises
        ------
        ValueError
            If the pipeline step is not found in the
            sofia_redux.instruments.hawc.steps package, or it cannot
            be imported.
        IOError
            If an input file cannot be read.
        """
        # check for valid input
        if len(self.input) == 0:
            raise ValueError("No input data to process.")

        # get step name from recipe
        try:
            if step_name is None:
                step_name = self.recipe[self.step_index]
            step_mod = 'step' + step_name
            step_class = self.step_class[step_name]
        except (IndexError, KeyError):
            raise ValueError("Pipeline step {} not found".format(step_name))

        # import the step module and class
        try:
            redmod = importlib.import_module(
                'sofia_redux.instruments.hawc.steps.{}'.format(step_mod))
            redclass = getattr(redmod, step_class)
        except (ImportError, AttributeError):
            raise ValueError("Pipeline step {} not found".format(step_name))

        # instantiate the class
        step = redclass()

        # get parameters and pass them to the step
        parset = self.get_parameter_set()
        pardict = {}
        if use_param:
            for pkey, pval in parset.items():
                pardict[pkey] = pval['value']

        # check whether output data should be saved, displayed, loaded
        try:
            save = parset['save']['value']
        except KeyError:
            save = False
        try:
            display = parset['display']['value']
        except KeyError:
            display = False
        try:
            load_data = parset['load']['value']
        except KeyError:
            load_data = True
        try:
            keep_auxout = parset['keep_auxout']['value']
        except KeyError:
            keep_auxout = False

        # first load the data if necessary

        for df in self.input:
            if load_data:
                if not df.loaded:
                    input_fname = df.filename
                    if os.path.isfile(df.filename):
                        fn = df.filename
                    elif os.path.isfile(df.rawname):
                        fn = df.rawname
                    else:
                        msg = 'Could not load ' \
                              'file {}'.format(df.filename)
                        log.error(msg)
                        raise IOError(msg)

                    log.debug('Loading {} into memory'.format(fn))
                    df.load(fn)

                    # reset filename to previous value -- DRP
                    # uses it to form output names for the next steps
                    df.filename = input_fname

            # set filename path to output directory after load
            # so that output goes to the right place
            df.filename = os.path.join(
                self.output_directory,
                os.path.basename(df.filename))

        # multi-input steps: require list input
        if step.iomode.startswith('MI'):
            log.info("Input: All files")

            # run the step
            newout = step(self.input, **pardict)

            # record/save the output
            if type(newout) is not list:
                newout = [newout]

            out = []
            auxout = []
            for df in newout:
                if save:
                    df.filename = os.path.join(
                        self.output_directory,
                        os.path.basename(df.filename))
                    df.save()
                    self.record_outfile(df.filename)

                out.append(df)

            # check for auxiliary out files
            if len(step.auxout) > 0:
                auxout += step.auxout

            # whitespace, for readability
            log.info('')
        else:
            # single input steps: loop over input
            out = []
            auxout = []
            for df in self.input:
                log.info("Input: {}".format(os.path.basename(df.filename)))

                # run the step
                try:
                    out_df = step(df, **pardict)
                except ValueError as err:
                    # if one file had an error, attempt
                    # to continue on with the rest
                    log.debug(err)
                    log.warning('Processing error: skipping file '
                                '{} in further reduction '
                                'steps'.format(
                                    os.path.basename(df.filename)))
                    continue

                # record/save the output
                if save:
                    out_df.filename = os.path.join(
                        self.output_directory,
                        os.path.basename(out_df.filename))
                    out_df.save()
                    self.record_outfile(out_df.filename)

                out.append(out_df)

                # check for auxiliary out files
                if len(step.auxout) > 0:
                    auxout += step.auxout

                log.info('')

        # keep the output data for the next step
        self.input = out

        # allow undo for the next step if possible
        try:
            next_parset = self.get_parameter_set(self.step_index + 1)
            self.allow_undo = next_parset['undo']['value']
        except (KeyError, IndexError):
            # leave as is if not available
            pass

        # pass self.input data to viewer
        if not keep_auxout:
            self.auxout = []
        self.set_display_data(display=display, auxout=auxout)

    def set_display_data(self, display=False, auxout=None):
        """
        Pass data in self.input to display.

        Parameters
        ----------
        display : bool, optional
            If True, data should be displayed in DS9.  If False,
            only the headers and `auxout` are displayed
        auxout : `list` of str, optional
            If provided, will be passed to the QADViewer along with
            any headers or data the input attribute.
        """
        if auxout is None:
            auxout = []
        else:
            auxout = auxout.copy()
        auxout += self.auxout[:]
        ddata = []
        if display:
            # make an hdulist out of the datafits object,
            # with no tables attached
            for df in self.input:
                ddata.append(df.to_hdulist(save_tables=False))
        else:
            # always display headers
            for df in self.input:
                ddata.append(df.to_header_list())

        # always display aux out (images, etc.)
        if len(auxout) > 0:
            ddata += auxout
            self.auxout = auxout

        self.display_data['QADViewer'] = ddata

    def step(self, alt_method=None):
        """
        Run a reduction step.

        This method called the reduction step specified in the
        `recipe` attribute, at the current `step_index`.  If the
        step index is out of range for the recipe, this method will
        just return.

        For the HAWC pipeline, `redux.Reduction.step` is called with
        default parameters for any step that is defined locally. The
        parent method is called with ``alt_method='run_drp_step'`` for
        any step that is defined in the DRP package only.

        Parameters
        ----------
        alt_method : str, optional
            This parameter is ignored for HAWC reductions.

        Returns
        -------
        str
            An error message if the reduction step produced any
            errors; an empty string otherwise.
        """

        if self.step_index >= len(self.recipe):
            return 'No steps to run.'
        step_name = self.recipe[self.step_index]
        if hasattr(self, step_name):
            return super().step()
        else:
            return super().step(alt_method='run_drp_step')

    # define some super-steps: combinations of DRP steps
    # to run as one step

    def make_flats(self):
        """
        Make nod or nod-pol flats from INT_CAL files.

        This step is called before the main nod or nod-pol
        steps, when intcal files (FITS keyword CALMODE = INT_CAL)
        are loaded along with science files (CALMODE = UNKNOWN).
        It produces flat files in a location known to the science
        steps, for later use in the current reduction.

        This super-step calls all steps defined for 'mode_intcal'
        in the DRP configuration, on all input files with mode 'intcal'.
        At the end of the step, the input attribute is set to
        include all non-intcal files, for processing the normal
        science pipeline steps.
        """
        flat_input = []
        other = []
        for df in self.input:
            if str(df.mode).lower().strip() == 'intcal':
                flat_input.append(df)
            else:
                other.append(df)

        if len(flat_input) == 0:
            # do nothing
            log.warning("No flats found.")
            return

        # get flat steps from config file in first input file
        df = flat_input[0]
        flat_steps = df.config['mode_intcal']['stepslist']

        # check for DMD file: modify recipe if necessary
        # No other intermediate input types are allowed
        # for intcal files.
        try:
            prodtype = df.getheadval('PRODTYPE', errmsg=False)
        except KeyError:
            prodtype = 'unknown'
        prodtype = str(prodtype).strip().lower()
        if prodtype == 'demodulate':
            idx = flat_steps.index('StepDemodulate')

            # recipe is all steps after the input one
            flat_steps = flat_steps[idx + 1:]

            if len(flat_steps) == 0:
                raise ValueError("No steps to run for "
                                 "prodtype {}.".format(prodtype))

        # run the steps on the input flats
        self.input = flat_input
        parset = self.get_parameter_set()
        save = parset['save']['value']
        for i, step in enumerate(flat_steps):
            try:
                step_name = self.step_name[step]
            except KeyError:
                raise ValueError("Pipeline step {} "
                                 "not found".format(step))

            log.info("Sub-step: {}".format(self.processing_steps[step_name]))
            log.info('')

            # set save parameter for final step only
            if i == len(flat_steps) - 1:
                parset['save']['value'] = save
                use_param = True
            else:
                parset['save']['value'] = False
                use_param = False

            # call the regular runner on the step
            self.run_drp_step(step_name=step_name, use_param=use_param)

        # flats are done;
        # set aside the rest of the files for processing
        self.input = other

    def process_intcal(self):
        """
        Process INT_CAL files for skycal generation.

        This step is an alias for the `make_flats` step,
        given a separate name in order to assign different default
        parameters.
        """
        self.make_flats()

    def demodulate(self):
        """
        Demodulate chop-nod data.

        This super-step calls all steps up to and including the
        StepDemodulate step defined for the current pipeline mode
        in the DRP configuration.  The full set of steps is called
        on one file at a time, such that only one raw, un-demodulated
        data file is in memory at a time.  This helps improve
        performance and memory usage for the early pipeline steps.
        """
        if len(self.input) == 0:
            raise ValueError('No input data.')

        # get demod steps from config file in first input file
        df = self.input[0]

        # if intermediate, just run the DRP demodulate
        if self.intermediate:
            self.run_drp_step(step_name='demodulate')
            return

        try:
            mode_steps = df.config['mode_{}'.format(df.mode)]['stepslist']
        except KeyError:
            # mode not found -- just run the DRP version of
            # demodulate
            self.run_drp_step(step_name='demodulate')
            return
        try:
            dmd_idx = mode_steps.index('StepDemodulate')
        except ValueError:
            log.warning('No demodulate steps found.')
            return
        demod_steps = mode_steps[:dmd_idx + 1]

        # get parameters for this step
        # (copy because intermediate steps set current parameters
        # to defaults)
        parset = deepcopy(self.get_parameter_set())

        # copy the input datafits (headers only) to a new list
        input_data = self.input.copy()
        self.input = []

        # run all steps on each file in turn, so that only
        # one raw file is in memory at a time
        out = []
        for df in input_data:
            self.input = [df]
            for i, step in enumerate(demod_steps):
                try:
                    step_name = self.step_name[step]
                except KeyError:
                    raise ValueError("Pipeline step {} "
                                     "not found".format(step))

                log.info("Sub-step: {}".format(
                    self.processing_steps[step_name]))
                log.info('')

                # use user parameters for final step only
                if i == len(demod_steps) - 1:
                    log.debug('Setting user parameters '
                              'for {}'.format(step_name))

                    # deepcopy again because next file will again modify
                    # the current parameters on the early steps
                    self.set_parameter_set(deepcopy(parset))
                else:
                    # set step defaults for Redux control parameters
                    try:
                        log.debug('Setting default parameters '
                                  'for {}'.format(step_name))
                        par_function = getattr(self.parameters, step_name)
                        par_function(self.step_index)
                    except AttributeError:
                        log.warning('Parameters for step '
                                    '{} not found'.format(step_name))

                # call the regular runner on the step
                self.run_drp_step(step_name=step_name)

            # keep the final output file
            out.append(self.input[0])

            # trigger garbage collection after each file
            # (doesn't seem to have much impact, but can't hurt)
            gc.collect()

        # store the demodulated data for the next step
        self.input = out
        self.set_display_data()

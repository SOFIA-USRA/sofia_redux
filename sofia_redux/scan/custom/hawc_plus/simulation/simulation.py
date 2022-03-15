from abc import ABC
from astropy import units
from astropy.coordinates import Angle, FK5
from astropy.io import fits
from astropy.time import Time
import getpass
import numpy as np
import warnings

from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import (
    GeodeticCoordinates)
from sofia_redux.scan.configuration.dates import DateRange
from sofia_redux.scan.custom.sofia.simulation.aircraft import \
    AircraftSimulation
from sofia_redux.scan.simulation.scan_patterns.daisy import \
    daisy_pattern_equatorial
from sofia_redux.scan.simulation.scan_patterns.lissajous import \
    lissajous_pattern_equatorial
from sofia_redux.scan.simulation.source_models.simulated_source import (
    SimulatedSource)
from sofia_redux.scan.utilities.utils import get_int_list
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.integration import integration_numba_functions as int_nf
from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector


class HawcPlusSimulation(ABC):

    # FITS header keys specific to this simulation or used to override defaults
    sim_keys = {'CHPNOISE', 'SCNDRAD', 'SCNDPER', 'SCNDNOSC',
                'SRCTYPE', 'SRCSIZE', 'SRCAMP', 'SRCS2N',
                'JUMPCHAN', 'JUMPFRMS', 'JUMPSIZE',
                'SPECTEL1', 'OBSRA', 'OBSDEC', 'DATE-OBS', 'LON_STA',
                'LAT_STA'}

    data_column_definitions = {
        'FrameCounter': ('frames', '1K'),  # X
        'Timestamp': ('seconds', '1D'),  # X
        'FluxJumps': ('jumps', '#I'),
        'SQ1Feedback': (None, '#J'),
        'hwpCounts': ('counts', '1J'),  # X
        'Flag': ('flag', '1J'),  # X
        'AZ': ('degrees', '1D'),  # X
        'EL': ('degrees', '1D'),  # X
        'RA': ('hours', '1D'),  # X
        'DEC': ('degrees', '1D'),  # X
        'LST': ('hours', '1D'),  # X
        'SIBS_VPA': ('degrees', '1D'),  # X
        'TABS_VPA': ('degrees', '1D'),  # X
        'Chop_VPA': ('degrees', '1D'),  # X
        'LON': ('degrees', '1D'),  # X
        'LAT': ('degrees', '1D'),  # X
        'NonSiderealRA': ('hours', '1D'),  # X
        'NonSiderealDec': ('degrees', '1D'),  # X
        'sofiaChopR': ('volts', '1E'),
        'sofiaChopS': ('volts', '1E'),
        'PWV': ('um', '1D'),  # X
        'LOS': ('degrees', '1D'),  # X
        'ROLL': ('degrees', '1D')  # X
    }

    default_values = {
        'HEADSTAT': 'SIMULATED',
        'FILTER': '-1,-1,-1,-1',
        'MCEMAP': '0,2,1,-1',
        'DATASRC': 'astro',
        'OBSTYPE': 'object',
        'KWDICT': 'UNKNOWN',
        'AOR_ID': '99_9999_9',
        'PROCSTAT': 'Level_0',
        'DATAQUAL': 'Nominal',
        'PLANID': '99_9999',
        'DEPLOY': 'UNKNOWN',
        'FLIGHTLG': 9,
        'ORIGIN': 'SOFSCAN simulation',
        'OBSERVER': getpass.getuser(),
        'CREATOR': 'SOFSCAN simulation',
        'OPERATOR': getpass.getuser(),
        'TELESCOP': 'SOFIA 2.5m',
        'TELCONF': 'NASMYTH',
        'TELEQUI': 'j2000',
        'TSC-STAT': 'STAB_INERTIAL_ONGOING',
        'FBC-STAT': 'FBC_ON',
        'OBSRA': -9999.0,
        'OBSDEC': -9999.0,
        'EQUINOX': 2000.0,
        'TRACMODE': 'offset+inertial',
        'TRACERR': False,
        'INSTRUME': 'HAWC_PLUS',
        'DATATYPE': 'OTHER',
        'INSTCFG': 'TOTAL_INTENSITY',
        'INSTMODE': 'OTFMAP',
        'DETECTOR': 'HAWC',
        'DETSIZE': '64,40',
        'PIXSCAL': -9999.0,
        'SIBS_X': 15.5,
        'SIBS_Y': 19.5,
        'XFPI': 433.861908,
        'YFPI': 447.023258,
        'COLLROTR': 0.0,
        'COLLROTS': 0.0,
        'FILEGPID': 1,
        'CDELT1': -9999.0,
        'CDELT2': -9999.0,
        'SMPLFREQ': 203.2520325203252,
        'CMTFILE': 'UNKNOWN',
        'CALMODE': 'UNKNOWN',
        'INTCALV': -9999.0,
        'DIAG_HZ': -9999.0,
        'RPTFILT': 'UNKNOWN',
        'RPTPUPIL': 'UNKNOWN',
        'HAWC_VER': 'UNKNOWN',
        'TRKAOI': 'UNKNOWN',
    }

    default_comments = {
        'FILTER': 'rowstart,rowend,colstart,colend',
        'MCEMAP': 'MCEs mapped to r0,r1,t0,t1 arrays',
        'DATASRC': 'Data source',
        'OBSTYPE': 'Observation type',
        'SRCTYPE': 'Source type',
        'SRCSIZE': 'The FWHM of the simulated source',
        'KWDICT': 'SOFIA keyword dictionary version',
        'OBS_ID': 'SOFIA Observation identification',
        'OBJECT': 'Object name',
        'AOR_ID': 'Astronomical Observation Request ID',
        'PROCSTAT': 'Processing status',
        'HEADSTAT': 'Header status',
        'DATAQUAL': 'Data quality',
        'PLANID': 'Observing plan ID',
        'DEPLOY': 'Site deployment',
        'MISSN-ID': 'Mission ID',
        'FLIGHTLG': 'Flight leg',
        'ORIGIN': 'Origin of FITS file',
        'OBSERVER': 'Observer(s)',
        'CREATOR': 'File creation task',
        'OPERATOR': 'Telescope operator',
        'FILENAME': 'Name of host file',
        'DATE': 'Date of file creation',
        'DATE-OBS': 'UTC date of exposure start',
        'UTCSTART': 'UTC of exposure start',
        'UTCEND': 'UTC of exposure end',
        'WVZ_STA': 'Water vapor, integrated to zenith, observation start [um]',
        'WVZ_END': 'Water vapor, integrated to zenith, observation start [um]',
        'TEMP_OUT': 'Static air temperature outside aircraft [C]',
        'TEMPPRI1': 'Temperature of primary mirror [C]',
        'TEMPPRI2': 'Temperature of primary mirror [C]',
        'TEMPPRI3': 'Temperature of primary mirror [C]',
        'TEMPSEC1': 'Temperature of secondary [C]',
        'ALTI_STA': 'Aircraft pressure altitude, start of observation [feet]',
        'ALTI_END': 'Aircraft pressure altitude, end of observation [feet]',
        'AIRSPEED': 'True aircraft airspeed [knots]',
        'GRDSPEED': 'Aircraft ground speed [knots]',
        'LAT_STA': 'Aircraft latitude, start of observation [deg]',
        'LAT_END': 'Aircraft latitude, end of observation [deg]',
        'LON_STA': 'Aircraft longitude, start of observation [deg]',
        'LON_END': 'Aircraft longitude, end of observation [deg]',
        'HEADING': 'Aircraft true heading [deg]',
        'TRACKANG': 'Aircraft track angle [deg]',
        'TELESCOP': 'Telescope name',
        'TELCONF': 'Telescope configuration',
        'TELRA': 'SI Boresight RA (ICRS J2000) [hours]',
        'TELDEC': 'SI Boresight DEC (ICRS J2000) [deg]',
        'TELVPA': 'SI Boresight VPA (ICRS J2000) [deg]',
        'TELEQUI': 'Equinox of ERF coords(RA/Dec/VPA)',
        'LASTREW': 'Time of last rewind (UTC)',
        'FOCUS_ST': 'Telescope focus - SMA FCM t position, obs. start [um]',
        'FOCUS_EN': 'Telescope focus - SMA FCM t position, obs. end [um]',
        'TELEL':
            'Telescope elevation at obs. start - as returned by MCCS [deg]',
        'TELXEL': ('Telescope cross elevation at obs. start - as returned by '
                   'MCCS [deg]'),
        'TELLOS': 'Telescope LOS at obs. start - as returned by MCCS [deg]',
        'TSC-STAT': 'TASCU Status at observation end',
        'FBC-STAT': 'FBC Status at observation end',
        'OBSRA': 'Requested RA [hours]',
        'OBSDEC': 'Requested DEC [deg]',
        'EQUINOX': 'Coordinate equinox for OBSRA and OBSDEC [yr]',
        'ZA_START': 'Telescope zenith angle, observation start [deg]',
        'ZA_END': 'Telescope zenith angle, observation end [deg]',
        'TRACMODE': 'SOFIA tracking mode',
        'TRACERR': 'Tracking error flag',
        'CHOPPING': 'Chopping flag',
        'NODDING': 'Nodding flag',
        'DITHER': 'Dithering flag',
        'MAPPING': 'Mapping flag',
        'SCANNING': 'Scanning flag',
        'INSTRUME': 'Instrument',
        'DATATYPE': 'Data type',
        'INSTCFG': 'Instrument configuration (int., polar.)',
        'INSTMODE': 'Instrument observing mode (c2n, otfmap)',
        'MCCSMODE': 'MCCS SI Mode - hawc_plus.si_config.current_mode',
        'EXPTIME': 'On-source exposure time [s]',
        'SPECTEL1': 'HAWC filter setting: HAW_A, HAW_B, etc',
        'SPECTEL2': 'HAWC pupil setting: HAW_HWP_A, etc',
        'WAVECENT': 'Central wavelength of observation [um]',
        'DETECTOR': 'Detector name',
        'DETSIZE': 'Detector size',
        'PIXSCAL': 'Pixel scale [arcsec]',
        'SIBS_X': 'SI pixel location of boresight (X)',
        'SIBS_Y': 'SI pixel location of boresight (Y)',
        'CHPFREQ': 'Chop frequency [Hz]',
        'CHPPROF': 'Chopping profile: 2 or 3 point',
        'CHPSYM': 'Chopping symmetry: symmetric or asymmetric',
        'CHPAMP1': 'Chop amplitude 1 [arcsec]',
        'CHPAMP2': 'Chop amplitude 2 [arcsec]',
        'CHPCRSYS': 'MCCS coord sys for sky tip, tilt, and angle',
        'CHPANGLE': 'Calc angle in the sky_coord_sys ref frame [arcsec]',
        'CHPTIP': 'Calc tip in the sky_coord_sys ref frame [arcsec]',
        'CHPTILT': 'Calc tilt in the sky_coord_sys ref frame [arcsec]',
        'CHPPHASE': 'Chop phase [ms]',
        'CHPSRC': 'Chop sync src [external,internal]',
        'NODTIME': 'Nod time [s]',
        'NODN': 'Nod cycles',
        'NODSETL': 'Nod settle time [s]',
        'NODAMP': 'Nod amplitude on sky [arcsec]',
        'NODBEAM': 'Current nod beam position',
        'NODPATT': 'Nodding pattern, one cycle',
        'NODSTYLE': 'Chop/nod style',
        'NODCRSYS': 'Coordinate system for Nod angle',
        'NODANGLE': 'Nod angle [deg]',
        'DTHCRSYS': 'Coordinate system for dithering',
        'DTHXOFF': 'Dither offset in X for this file',
        'DTHYOFF': 'Dither offset in Y for this file',
        'DTHPATT': 'Dither pattern',
        'DTHNPOS': 'Number of dither positions',
        'DTHINDEX': 'Dither position index',
        'DTHUNIT': 'Dither units - pixel or arcsec',
        'DTHSCALE': 'Dither scale [float]',
        'SCNCRSYS': 'Scan coordinate system',
        'SCNRATE': 'Scan rate [arcsec/s]',
        'SCNITERS': 'Scan iterations',
        'SCNANGLS': 'Scan angle start (first iteration) [deg]',
        'SCNANGLC': 'Current scan angle [deg]',
        'SCNANGLF': 'Scan angle finish (last iteration) [deg]',
        'SCNTWAIT': 'Scan tracking measurement window [sec]',
        'SCNTRKON': 'Track continuously while scanning [0,1]',
        'SCNAMPEL': 'Lissajous scan amplitude in elevation [arcsec]',
        'SCNAMPXL': 'Lissajous scan amplitude in cross-elevation [arcsec]',
        'SCNDUR': 'Lissajous requested scan duration [sec]',
        'SCNFQRAT': 'Lissajous pattern frequency ratio',
        'SCNPHASE': 'Lissajous pattern relative phase offset [deg]',
        'SCNTOFF': 'Lissajous pattern relative time offset [sec]',
        'SCNNSUBS': 'Raster number of subscans',
        'SCNLEN': 'Raster length of single scan line [pix or arcsec]',
        'SCNSTEP':
            'Raster size of step from one linear scan to the next [pix or',
        'SCNSTEPS': 'Raster number of linear scan lines',
        'SCNCROSS': 'Raster scan includes paired cross-scan?',
        'XFPI': 'FPI column for the Position or AOI chop image - coord.pos',
        'YFPI': 'FPI row for the Position or AOI chop image - coord.pos.si',
        'COLLROTR':
            'Collimation position r - ta_scs.fcm_status.fcm_des_coll_rot_r',
        'COLLROTS':
            'Collimation position s - ta_scs.fcm_status.fcm_des_coll_rot_s',
        'BSITE': 'boresight name - hawc_plus.si_config.current_mode',
        'OBSMODE': 'Observation mode [Scan, ChopScan, DitherChopNod]',
        'SCNPATT': 'Scan pattern [Raster, Lissajous, Daisy]',
        'FILEGPID': 'File group ID',
        'OBJRA': 'coord.pos.target RA [hours]',
        'OBJDEC': 'coord.pos.target DEC [deg]',
        'CDELT1': 'Plate scale for the n-th axis at ref pnt [deg/pix]',
        'CDELT2': 'Plate scale for the n-th axis at ref pnt [deg/pix]',
        'NHWP': 'Number of HWP angles',
        'SMPLFREQ': 'Sampling frequency [Hz]',
        'HWPSTART': 'HWP initial angle [deg]',
        'HWPSPEED': 'HWP speed',
        'HWPSTEP': 'HWP step [deg]',
        'HWPSEQ': 'HWP list of angles [degs]',
        'CHPONFPA': 'Is chop on-chip or off-chip.',
        'CMTFILE': 'MCCS Comment Filename',
        'CALMODE': 'Diagnostic procedure mode',
        'INTCALV': 'INT-CAL voltage',
        'DIAG_HZ': 'Diagnostic procedure chop rate (if used) [hz]',
        'RPTFILT': 'Reported filter position from OMS',
        'RPTPUPIL': 'Reported pupil position from OMS',
        'HWPON': 'ON HWP threshold [volts]',
        'HWPOFF': 'OFF HWP threshold [volts]',
        'HWPHOME': 'HOME HWP threshold [volts]',
        'HAWC_VER': 'HAWC+ CDH Software Version',
        'TOTTIME': 'Total archiving time [sec]',
        'TRKAOI': 'Tracking AOI',
        'FCSCOEFA': 'focus_coef_a',
        'FCSCOEFB': 'focus_coef_b',
        'FCSCOEFC': 'focus_coef_c',
        'FCSCOEFK': 'focus_coef_k',
        'FCSCOEFQ': 'focus_coef_q',
        'FCSDELTA': 'focus_delta',
        'FCSTCALC': 'focus_fcm_t_calc',
        'FCST1NM': 'focus_param_t1',
        'FCST2NM': 'focus_param_t2',
        'FCST3NM': 'focus_param_t3',
        'FCSXNM': 'focus_param_x',
        'FCST1': 'focus_param_value_t1',
        'FCST2': 'focus_param_value_t2',
        'FCST3': 'focus_param_value_t3',
        'FCSX': 'focus_param_value_x',
        'FCSTOFF': 'focus_total_offset',
        'SDELSTEN': 'Skydip elevation at start and end [deg]',
        'SDELMID': 'Skydip elevation at middle [deg]',
        'SDWTSTRT': 'Skydip wait at start [sec]',
        'SDWTMID': 'Skydip wait at middle [sec]',
        'SDWTEND': 'Skydip wait at end [sec]',
        'DBRA0': 'Drift 0 begin RA [hours]',
        'DBDEC0': 'Drift 0 begin Dec [deg]',
        'DBTIME0': 'Drift 0 begin time [seconds]',
        'DARA0': 'Drift 0 after RA [hours]',
        'DADEC0': 'Drift 0 after Dec [deg]',
        'DATIME0': 'Drift 0 after time [seconds]',
        'XPADDING': 'for header size changes',
        'MAPCRSYS': 'Mapping coordinate system',
        'MAPNXPOS': 'Number of map positions in X',
        'MAPNYPOS': 'Number of map positions in Y',
        'MAPINTX': 'Map step interval in X [arcmin]',
        'MAPINTY': 'Map step interval in Y [arcmin]',
        'SCNDRAD': 'Daisy scan radius [arcsec]',
        'SCNDPER': 'Daisy scan radial period [seconds]',
        'SCNDNOSC': 'Daisy scan number of oscillations',
    }

    def __init__(self, info):
        """
        Initialize a HAWC+ simulation.

        Parameters
        ----------
        info : HawcPlusInfo
        """
        if not isinstance(info, HawcPlusInfo):
            raise ValueError(
                f"Simulation must be initialized with {HawcPlusInfo}")
        self.info = info
        self.hdul = None
        self.user = getpass.getuser()
        self.aircraft = AircraftSimulation()
        self.source_equatorial = None
        self.start_utc = None
        self.end_utc = None
        self.start_site = None
        self.primary_header = fits.Header()
        self.channels = info.get_channels_instance()
        self.channels.set_parent(self)
        self.scan = None
        self.integration = None
        self.column_values = None
        self.equatorial = None
        self.apparent_equatorial = None
        self.horizontal = None
        self.equatorial_corrected = None  # without chopper
        self.horizontal_corrected = None  # without chopper
        self.horizontal_offset_corrected = None  # without chopper
        self.apparent_equatorial_corrected = None  # without chopper
        self.horizontal_offset = None
        self.lst = None
        self.mjd = None
        self.site = None
        self.sin_pa = None
        self.cos_pa = None
        self.chopper_position = None
        self.source_model = None
        self.source_data = None
        self.source_max = None
        self.data_hdu = None
        self.projection = None
        self.projector = None
        self.model_offsets = None

    @classmethod
    def default_value(cls, key):
        """
        Return the default value for a given key.

        Parameters
        ----------
        key : str
           The key for which to retrieve a default value.  If 'all', then all
           default values will be returned.

        Returns
        -------
        value : int or float or str or bool or dict
        """

        if key == 'all':
            return cls.default_values.copy()
        return cls.default_values.get(key.strip().upper())

    @classmethod
    def default_comment(cls, key):
        """
        Return the default comment for a given key.

        Parameters
        ----------
        key : str

        Returns
        -------
        comment : str
        """
        if key == 'all':
            return cls.default_comments.copy()
        return cls.default_comments.get(key.strip().upper(), '')

    def update_header_value(self, header, key, value):
        """
        Update the header with a new value.

        Parameters
        ----------
        header : fits.Header
        key : str
        value : str or int or float or bool

        Returns
        -------
        None
        """
        if key not in header:
            header[key] = value, self.default_comment(key)
        else:
            comment = header.comments[key]
            if comment == '':
                comment = self.default_comment(key)
            header[key] = value, comment

    def write_simulated_hdul(self, filename,
                             ra='17h45m39.60213s',
                             dec='-29d00m22.0000s',
                             site_longitude='-122.0644d',
                             site_latitude='37.4089d',
                             date_obs='2021-12-06T18:48:25.876',
                             overwrite=True,
                             header_options=None):
        """
        Write a simulated HDU list to file.

        Parameters
        ----------
        filename : str
            The file path to the output file for which to write the simulated
            HDU list.
        ra : str or units.Quantity, optional
            The right-ascension of the simulated source.  The default is the
            Galactic center.
        dec : str or units.Quantity, optional
            The declination of the simulated source.  The default is the
            Galactic center.
        site_longitude : str or units.Quantity, optional
            The site longitude of the simulated observation.  The default is
            NASA Ames.
        site_latitude : str or units.Quantity, optional
            The site latitude of the simulated observation.  The default is
            NASA Ames.
        date_obs : str or Time, optional
            The date of the simulated observation.  String values should be
            provided in ISOT format in UTC scale.
        overwrite : bool, optional
            If `True`, allow `filename` to be overwritten if it already exists.
        header_options : fits.Header or dict, optional
            Optional settings to add to the primary header.

        Returns
        -------
        None
        """
        if header_options is None:
            header_options = dict()
        hdul = self.create_simulated_hdul(ra=ra, dec=dec,
                                          site_longitude=site_longitude,
                                          site_latitude=site_latitude,
                                          date_obs=date_obs,
                                          header_options=header_options)
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()

    def create_simulated_hdul(self,
                              ra='17h45m39.60213s',
                              dec='-29d00m22.0000s',
                              site_longitude='-122.0644d',
                              site_latitude='37.4089d',
                              date_obs='2021-12-06T18:48:25.876',
                              header_options=None):
        """
        Create an HDU list containing simulated data.

        The simulated HDU list contains the following data:

          - hdul[0] = PrimaryHDU
              data : (0,) float32
              header : The primary header
          - hdul[1] = ImageHDU
              data : (1,) int32
              header : The MCE configuration.  Not used in the reduction.
          - hdul[2] = BinTableHDU
              data : A FITS table containing the data used in the reduction.
              header : Description of the data in the FITS table.

        Parameters
        ----------
        ra : str or units.Quantity, optional
            The right-ascension of the simulated source.  The default is the
            Galactic center.
        dec : str or units.Quantity, optional
            The declination of the simulated source.  The default is the
            Galactic center.
        site_longitude : str or units.Quantity, optional
            The site longitude of the simulated observation.  The default is
            NASA Ames.
        site_latitude : str or units.Quantity, optional
            The site latitude of the simulated observation.  The default is
            NASA Ames.
        date_obs : str or Time, optional
            The date of the simulated observation.  String values should be
            provided in ISOT format in UTC scale.
        header_options : fits.Header or dict, optional
            Optional settings to add to the primary header.

        Returns
        -------
        hdul : fits.HDUList
            A simulated FITS HDU list.
        """
        self.create_basic_hdul(ra=ra, dec=dec, site_longitude=site_longitude,
                               site_latitude=site_latitude, date_obs=date_obs,
                               header_options=header_options)

        self.update_hdul_with_data()
        self.update_primary_header_with_data_hdu()
        return self.hdul

    def update_primary_header_with_data_hdu(self):
        """
        Fill in the missing values in the primary header using data values.

        Returns
        -------
        None
        """
        h = self.primary_header
        if 'TELVPA' not in h:
            vpa = self.column_values['TABS_VPA'][0]
            self.update_header_value(h, 'TELVPA', vpa)

        if 'TELEL' not in h or 'TELXEL' not in h:
            source_horizontal = self.source_equatorial.to_horizontal(
                self.site, self.lst)
            offset = self.horizontal.get_native_offset_from(source_horizontal)
            if 'TELEL' not in h:
                el = source_horizontal.el[0].to('degree').value
                self.update_header_value(h, 'TELEL', el)
            if 'TELXEL' not in h:
                xel = offset[0].x.to('degree').value
                self.update_header_value(h, 'TELXEL', xel)

        if 'TELLOS' not in h:  # TODO: This is wrong, but not used.
            self.update_header_value(h, 'TELLOS', -h['TELXEL'])

        za = self.integration.frames.horizontal.za.to('degree').value
        if 'ZA_START' not in h:
            self.update_header_value(h, 'ZA_START', za[0])
        if 'ZA_END' not in h:
            self.update_header_value(h, 'ZA_END', za[-1])
        self.hdul[0].header = h

    def create_basic_hdul(self,
                          ra='17h45m39.60213s',
                          dec='-29d00m22.0000s',
                          site_longitude='-122.0644d',
                          site_latitude='37.4089d',
                          date_obs='2021-12-06T18:48:25.876',
                          header_options=None):
        """
        Create a basic HDU list containing the basic primary header info.

        Creates the HDUs containing the necessary information which may be
        used to create a final data HDU.

        Parameters
        ----------
        ra : str or units.Quantity, optional
            The right-ascension of the simulated source.  The default is the
            Galactic center.
        dec : str or units.Quantity, optional
            The declination of the simulated source.  The default is the
            Galactic center.
        site_longitude : str or units.Quantity, optional
            The site longitude of the simulated observation.  The default is
            NASA Ames.
        site_latitude : str or units.Quantity, optional
            The site latitude of the simulated observation.  The default is
            NASA Ames.
        date_obs : str or Time, optional
            The date of the simulated observation.  String values should be
            provided in ISOT format in UTC scale.
        header_options : fits.Header or dict, optional
            Optional settings to add to the primary header.

        Returns
        -------
        None
        """
        self.hdul = fits.HDUList()
        primary_header = self.create_primary_header(
            ra=ra, dec=dec,
            site_latitude=site_latitude, site_longitude=site_longitude,
            date_obs=date_obs, header_options=header_options)

        primary_hdu = fits.PrimaryHDU(
            header=primary_header, data=np.empty(0, dtype=np.float32))
        self.primary_header = primary_hdu.header
        self.hdul.append(primary_hdu)

        config_hdu = self.create_configuration_hdu()
        self.hdul.append(config_hdu)

    def create_primary_header(self,
                              ra='17h45m39.60213s',
                              dec='-29d00m22.0000s',
                              site_longitude='-122.0644d',
                              site_latitude='37.4089d',
                              date_obs='2021-12-06T18:48:25.876',
                              header_options=None):
        """
        Create a simulated observation primary header.

        Parameters
        ----------
        ra : str or units.Quantity, optional
            The right-ascension of the observed source.
        dec : str or units.Quantity, optional
            The declination of the observed source.
        site_longitude : str or units.Quantity, optional
            The site longitude at the start of the observation.
        site_latitude : str or units.Quantity, optional
            The site latitude at the start of the observation.
        date_obs : str, optional
            The date-time of the observation in ISOT format, UTC scale at the
            start of the observation.
        header_options : fits.Header or dict, optional
            Optional header keywords for inclusion in the primary header.

        Returns
        -------
        header : fits.Header
        """
        self.primary_header = self.default_primary_header(header_options)
        self.update_header_band(self.primary_header)
        self.set_source(ra=ra, dec=dec, header_options=header_options)
        self.update_header_chopping(self.primary_header)
        self.update_header_nodding(self.primary_header)
        self.update_header_dithering(self.primary_header)
        self.update_header_mapping(self.primary_header)
        self.update_header_scanning(self.primary_header)
        self.update_header_hwp(self.primary_header)
        self.update_header_focus(self.primary_header)
        self.update_header_skydip(self.primary_header)
        self.set_times(date_obs, header_options=header_options)
        self.set_start_site(site_longitude, site_latitude,
                            header_options=header_options)
        self.initialize_aircraft()
        self.update_header_weather(self.primary_header)
        self.update_header_origin(self.primary_header)
        self.create_source_model()

        return self.primary_header

    def default_primary_header(self, header_options):
        """
        Create a default primary header.

        Parameters
        ----------
        header_options : fits.Header or dict or None
            The initial header with which to update and apply any defaults.

        Returns
        -------
        fits.Header
        """
        h = fits.Header()
        for key, value in self.default_value('all').items():
            self.update_header_value(h, key, value)

        if isinstance(header_options, fits.Header):
            h.update(header_options)
        elif isinstance(header_options, dict):
            for key, value in header_options.items():
                if key not in h or isinstance(value, tuple):
                    h[key] = value
                else:
                    self.update_header_value(h, key, value)
        return h

    def update_header_band(self, header):
        """
        Update a given header with HAWC band information.

        Parameters
        ----------
        header : fits.Header
            The header to update.

        Returns
        -------
        None
        """
        centers = {'A': 53.0, 'B': 63.0, 'C': 89.0, 'D': 154.0, 'E': 214.0}

        spec = header.get('SPECTEL1', 'HAW_A').split('_')
        if len(spec) <= 1:
            band = 'A'
        else:
            band = spec[-1].strip().upper()

        key = 'MCCSMODE'
        if key not in header:
            self.update_header_value(
                header, key, f'band_{band.lower()}_foctest')

        key = 'SPECTEL1'
        if key not in header:
            self.update_header_value(header, key, f'HAW_{band}')

        key = 'SPECTEL2'
        if key not in header:
            self.update_header_value(header, key, 'HAW_HWP_Open')

        key = 'WAVECENT'
        if key not in header:
            self.update_header_value(header, key, centers.get(band, -9999.0))

        key = 'BSITE'
        if key not in header:
            self.update_header_value(
                header, key, f'band_{band.lower()}_foctest')

    def update_header_origin(self, header):
        """
        Update the origin-like parameters in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        date = header.get('DATE-OBS', self.start_utc.isot).split('T')[0]
        prefix = f'{date}_HA_F999'
        plan = ''.join(header['PLANID'].split('_'))

        if 'OBS_ID' not in header:
            self.update_header_value(header, 'OBS_ID', f'{prefix}-sim-999')
        if 'MISSN-ID' not in header:
            self.update_header_value(header, 'MISSN-ID', prefix)
        if 'FILENAME' not in header:
            band = header['SPECTEL1'].split('_')[-1]
            self.update_header_value(
                header, 'FILENAME',
                f'{prefix}_999_SIM_{plan}_HAW{band}_RAW.fits')

    def update_header_chopping(self, header):
        """
        Update the chopping configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'CHOPPING': False,
            'CHPFREQ': 10.2,
            'CHPPROF': '2-POINT',
            'CHPSYM': 'no_chop',
            'CHPAMP1': 0.0,
            'CHPAMP2': 0.0,
            'CHPCRSYS': 'tarf',
            'CHPANGLE': 0.0,
            'CHPTIP': 0.0,
            'CHPTILT': 0.0,
            'CHPSRC': 'external',
            'CHPONFPA': False
        }
        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_nodding(self, header):
        """
        Update the nodding configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'NODDING': False,
            'NODTIME': -9999.0,
            'NODN': 1,
            'NODSETL': -9999.0,
            'NODAMP': 150.0,
            'NODBEAM': 'a',
            'NODPATT': 'ABBA',
            'NODSTYLE': 'NMC',
            'NODCRSYS': 'erf',
            'NODANGLE': -90.0}
        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_dithering(self, header):
        """
        Update the dithering configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'DITHER': False,
            'DTHCRSYS': 'UNKNOWN',
            'DTHXOFF': -9999.0,
            'DTHYOFF': -9999.0,
            'DTHPATT': 'NONE',
            'DTHNPOS': -9999,
            'DTHINDEX': -9999,
            'DTHUNIT': 'UNKNOWN',
            'DTHSCALE': -9999.0}

        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_mapping(self, header):
        """
        Update the mapping configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'MAPPING': False,
            'MAPCRSYS': 'UNKNOWN',
            'MAPNXPOS': -9999,
            'MAPNYPOS': -9999,
            'MAPINTX': -9999.0,
            'MAPINTY': -9999.0}

        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_hwp(self, header):
        """
        Update the half-wave-plate configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'NHWP': 1,
            'HWPSTART': -9999.0,
            'HWPSPEED': -9999,
            'HWPSTEP': -9999.0,
            'HWPSEQ': 'UNKNOWN',
            'HWPON': 10.0,
            'HWPOFF': 9.0,
            'HWPHOME': 8.0}

        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_focus(self, header):
        """
        Update the focus configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'FOCUS_ST': 800.0,
            'FOCUS_EN': 800.0,
            'FCSCOEFA': -13.8,
            'FCSCOEFB': 0.0,
            'FCSCOEFC': 0.0,
            'FCSCOEFK': -4.23,
            'FCSCOEFQ': 0.0,
            'FCSDELTA': 381.569916,
            'FCSTCALC': 800.0,
            'FCST1NM': 'ta_mcp.mcp_hk_pms.tmm_temp_1',
            'FCST2NM': '',
            'FCST3NM': '',
            'FCSXNM': '',
            'FCST1': -27.649994,
            'FCST2': -9999.0,
            'FCST3': -9999.0,
            'FCSX': -9999.0,
            'FCSTOFF': 25.0
        }

        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_skydip(self, header):
        """
        Update the skydip configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        defaults = {
            'SDELSTEN': -9999.0,
            'SDELMID': -9999.0,
            'SDWTSTRT': -9999.0,
            'SDWTMID': -9999.0,
            'SDWTEND': -9999.0}

        for key, value in defaults.items():
            if key not in header:
                self.update_header_value(header, key, value)

    def update_header_scanning(self, header):
        """
        Update the skydip configuration in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        if 'EXPTIME' in header:
            scan_length = header['EXPTIME']
        elif 'TOTTIME' in header:
            scan_length = header['TOTTIME']
        else:
            scan_length = 30.0

        for (key, default) in [
                ('EXPTIME', scan_length), ('TOTTIME', scan_length),
                ('SCANNING', True), ('OBSMODE', 'Scan'), ('SCNPATT', 'Daisy'),
                ('SCNCRSYS', 'TARF'), ('SCNITERS', 1), ('SCNANGLS', 0.0),
                ('SCNANGLC', 0.0), ('SCNANGLF', 0.0), ('SCNTWAIT', 0.0),
                ('SCNTRKON', 0), ('SCNRATE', 100.0)]:
            if key not in header:
                self.update_header_value(header, key, default)

        scan_pattern = header.get('SCNPATT').lower().strip()
        if scan_pattern not in ['daisy', 'lissajous']:
            raise ValueError(f"{header['SCNPATT']} scanning pattern is not "
                             f"currently supported.")

        self.update_header_lissajous(header)
        self.update_header_daisy(header)
        self.update_header_raster(header)

    def update_header_lissajous(self, header):
        """
        Update the Lissajous scanning parameters in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        if header['SCNPATT'].strip().lower() != 'lissajous':
            for key in ['SCNAMPEL', 'SCNAMPXL', 'SCNFQRAT', 'SCNPHASE',
                        'SCNTOFF']:
                self.update_header_value(header, key, -9999.0)
            return

        source_size = header['SRCSIZE']
        extended = header['SRCTYPE'].lower().strip() == 'extended'
        x = y = None
        if 'SCNAMPXL' in header:
            x = float(header['SCNAMPXL'])
        if 'SCNAMPEL' in header:
            y = float(header['SCNAMPEL'])
        if x is None and y is not None:
            x = y
        elif y is None and x is not None:
            y = x
        else:
            width = source_size * 5
            if extended:
                width /= 2
            x = y = width

        self.update_header_value(header, 'SCNAMPXL', x)
        self.update_header_value(header, 'SCNAMPEL', y)
        if 'SCNFQRAT' not in header:
            self.update_header_value(header, 'SCNFQRAT', np.sqrt(2))
        if 'SCNPHASE' not in header:
            self.update_header_value(header, 'SCNPHASE', 90.0)
        if 'SCNTOFF' not in header:
            self.update_header_value(header, 'SCNTOFF', 0.0)

    def update_header_daisy(self, header):
        """
        Update the daisy scanning parameters in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        if header['SCNPATT'].strip().lower() != 'daisy':
            for key in ['SCNDRAD', 'SCNDPER', 'SCNDNOSC']:
                self.update_header_value(header, key, -9999.0)
            return

        if 'SCNDRAD' not in header:
            source_size = header['SRCSIZE']
            extended = header['SRCTYPE'].lower().strip() == 'extended'
            if extended:
                radius = source_size * 2.5
            else:
                radius = source_size * 5
            self.update_header_value(header, 'SCNDRAD', radius)

        scan_length = header['EXPTIME']

        if 'SCNDPER' not in header and 'SCNDNOSC' not in header:
            self.update_header_value(header, 'SCNDNOSC', 22.0)
        if 'SCNDPER' not in header:
            n_oscillations = header['SCNDNOSC']
            radial_period = scan_length / n_oscillations
        elif 'SCNDNOSC' not in header:
            radial_period = header['SCNDPER']
            n_oscillations = scan_length / radial_period
        else:
            radial_period = header['SCNDPER']
            n_oscillations = header['SCNDNOSC']
        self.update_header_value(header, 'SCNDNOSC', n_oscillations)
        self.update_header_value(header, 'SCNDPER', radial_period)

    def update_header_raster(self, header):
        """
        Update the raster scanning parameters in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        for (key, value) in [('SCNNSUBS', -9999), ('SCNLEN', -9999.0),
                             ('SCNSTEP', -9999.0), ('SCNSTEPS', -9999.0),
                             ('SCNCROSS', False)]:
            self.update_header_value(header, key, value)

    def set_source(self, ra, dec, header_options=None):
        """
        Set the source RA and Dec coordinates.

        Parameters
        ----------
        ra : str or units.Quantity
        dec : str or units.Quantity
        header_options : dict or fits.Header, optional
            Optional header options.  The OBSRA and OBSDEC keys will override
            ra and dec if present.

        Returns
        -------
        None
        """
        if header_options is not None:
            if 'OBSRA' in header_options:
                ra = header_options['OBSRA']
                if not isinstance(ra, units.Quantity):
                    ra = ra * units.Unit('hourangle')
            if 'OBSDEC' in header_options:
                dec = header_options['OBSDEC']
                if not isinstance(dec, units.Quantity):
                    dec = dec * units.Unit('degree')

        if isinstance(ra, str):
            center = FK5(ra=ra, dec=dec)
            center = EquatorialCoordinates([center.ra, center.dec])
        else:
            center = EquatorialCoordinates([ra, dec])
        self.source_equatorial = center
        ra_value = center.ra.to('hourangle').value
        dec_value = center.dec.to('degree').value
        self.update_header_value(self.primary_header, 'TELRA', ra_value)
        self.update_header_value(self.primary_header, 'OBJRA', ra_value)
        self.update_header_value(self.primary_header, 'TELDEC', dec_value)
        self.update_header_value(self.primary_header, 'OBJDEC', dec_value)
        if 'OBJECT' not in self.primary_header:
            self.update_header_value(self.primary_header, 'OBJECT',
                                     'simulated_source')

        if 'SRCTYPE' not in self.primary_header:
            self.update_header_value(self.primary_header, 'SRCTYPE',
                                     'point_source')
        source_type = self.primary_header['SRCTYPE'].strip().lower()

        if 'SRCSIZE' not in self.primary_header:
            arcsec_fwhms = {
                'A': 4.85, 'B': 10.5, 'C': 7.8, 'D': 13.6, 'E': 18.2}
            band = self.primary_header['SPECTEL1'].split('_')[-1]
            fwhm = arcsec_fwhms.get(band)
            if source_type == 'extended':
                fwhm *= 3
            self.update_header_value(self.primary_header, 'SRCSIZE', fwhm)

    def set_times(self, timestamp, header_options=None):
        """
        Set the times for the observation.

        Parameters
        ----------
        timestamp : str or int or float or Time
            The object to convert.   If a string is used, it should be in ISOT
            format in UTC scale.  Integers and floats will be parsed as MJD
            times in the UTC scale.
        header_options : dict or fits.Header, optional
            An optional set of keyword values that will override `timestamp`
            with the 'DATE-OBS' key value if present.

        Returns
        -------
        None
        """
        if header_options is not None:
            if 'DATE-OBS' in header_options:
                timestamp = header_options['DATE-OBS']

        self.start_utc = DateRange.to_time(timestamp)
        self.info.astrometry.set_mjd(self.start_utc.mjd)
        self.info.astrometry.calculate_precessions(J2000)

        date_obs = self.start_utc.isot
        self.update_header_value(self.primary_header, 'DATE-OBS', date_obs)
        self.update_header_value(self.primary_header, 'DATE', date_obs)
        self.update_header_value(self.primary_header, 'UTCSTART',
                                 date_obs.split('T')[-1])
        scan_length = self.primary_header['EXPTIME'] * units.Unit('second')
        self.end_utc = self.start_utc + scan_length
        self.update_header_value(self.primary_header, 'UTCEND',
                                 self.end_utc.isot.split('T')[-1])

        if 'LASTREW' not in self.primary_header:
            last_rew = self.start_utc - (1 * units.Unit('minute'))
            self.update_header_value(self.primary_header, 'LASTREW',
                                     last_rew.isot)

    def set_start_site(self, longitude, latitude,
                       header_options=None):
        """
        Set the site longitude and latitude at the start of the observation.

        Parameters
        ----------
        longitude : str or units.Quantity
        latitude : str or units.Quantity
        header_options : dict or fits.Header, optional
            If provided, the LON_STA and LAT_STA will override `longitude` and
            `latitude` if present.

        Returns
        -------
        None
        """
        if header_options is not None:
            if 'LAT_STA' in header_options:
                latitude = header_options['LAT_STA']
                if not isinstance(latitude, units.Quantity):
                    latitude = latitude * units.Unit('degree')
            if 'LON_STA' in header_options:
                longitude = header_options['LON_STA']
                if not isinstance(longitude, units.Quantity):
                    longitude = longitude * units.Unit('degree')

        if isinstance(latitude, str):
            latitude = Angle(latitude)
            longitude = Angle(longitude)

        site = GeodeticCoordinates([longitude, latitude], unit='degree')
        self.update_header_value(self.primary_header, 'LON_STA',
                                 site.longitude.value)
        self.update_header_value(self.primary_header, 'LAT_STA',
                                 site.latitude.value)
        self.start_site = site

    def initialize_aircraft(self):
        """
        Set the aircraft parameters.

        Returns
        -------
        None
        """
        self.aircraft.initialize_from_header(self.primary_header)
        self.update_header_value(self.primary_header, 'ALTI_STA',
                                 self.aircraft.start_altitude.value)
        self.update_header_value(self.primary_header, 'ALTI_END',
                                 self.aircraft.end_altitude.value)
        self.update_header_value(self.primary_header, 'AIRSPEED',
                                 self.aircraft.airspeed.value)
        self.update_header_value(self.primary_header, 'GRDSPEED',
                                 self.aircraft.ground_speed.value)
        self.update_header_value(self.primary_header, 'LON_END',
                                 self.aircraft.end_location.longitude.value)
        self.update_header_value(self.primary_header, 'LAT_END',
                                 self.aircraft.end_location.latitude.value)
        self.update_header_value(self.primary_header, 'HEADING',
                                 self.aircraft.heading.value)
        self.update_header_value(self.primary_header, 'TRACKANG',
                                 self.aircraft.heading.value)

    def update_header_weather(self, header):
        """
        Update the weather parameters in the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        t = -47.0
        if 'TEMP_OUT' not in header:
            self.update_header_value(header, 'TEMP_OUT', t)
        if 'TEMPPRI1' not in header:
            self.update_header_value(header, 'TEMPPRI1', t + 34)
        if 'TEMPPRI2' not in header:
            self.update_header_value(header, 'TEMPPRI2', t + 32)
        if 'TEMPPRI3' not in header:
            self.update_header_value(header, 'TEMPPRI3', t + 33)
        if 'TEMPSEC1' not in header:
            self.update_header_value(header, 'TEMPSEC1', t + 30)

        if 'WVZ_STA' not in header or 'WVZ_END' not in header:
            pwv41k = self.info.configuration.get_float(
                'pwv41k', default=29.0)
            b = 1.0 / self.info.configuration.get_float('pwvscale',
                                                        default=5.0)

            for k in ['STA', 'END']:
                key = f'WVZ_{k}'
                if key not in header:
                    height = header[f'ALTI_{k}'] / 1000
                    self.update_header_value(
                        header, key, pwv41k * np.exp(-b * (height - 41.0)))

    def create_configuration_hdu(self):
        """
        Create the configuration HDU for the simulated data.

        Defines all bias lines as 5000 for subarrays 0->2.

        Returns
        -------
        fits.ImageHDU
        """
        bias_lines = self.info.detector_array.MCE_BIAS_LINES
        n_sub = self.info.detector_array.subarrays

        header = fits.Header()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fits.verify.VerifyWarning)
            for sub in range(n_sub - 1):  # Skip last subarray
                bias = ','.join(['1'] * bias_lines)
                key = f"MCE{sub}_TES_BIAS"
                header[key] = bias

        config_hdu = fits.ImageHDU(data=np.zeros(1, dtype=np.int32),
                                   header=header, name='Configuration')
        return config_hdu

    def update_hdul_with_data(self):
        """
        Create the data HDU.

        Returns
        -------
        None
        """
        self.scan = self.channels.get_scan_instance()
        self.scan.info.parse_header(self.hdul[0].header)
        self.scan.channels.read_data(self.hdul)
        self.scan.channels.validate_scan(self)
        self.scan.hdul = self.hdul
        self.integration = self.scan.get_integration_instance()
        self.update_non_astronomical_columns()
        n_records = self.column_values['FrameCounter'].size
        self.integration.frames.initialize(self.integration, n_records)
        self.update_vpa_columns()
        self.update_chopper()
        self.update_astronomical_columns()
        self.integration.frames.apply_hdu(self.get_data_hdu())
        self.create_simulated_data()
        self.create_simulated_jumps()
        self.data_hdu = self.get_data_hdu()
        self.hdul.append(self.data_hdu)

    def update_non_astronomical_columns(self):
        """
        Create column values for the FITS data HDU that are not astronomical.

        Returns
        -------
        None
        """
        header = self.primary_header
        dt = self.scan.info.sampling_interval
        scan_length = header['EXPTIME'] * units.Unit('second')
        n_records = int((scan_length / dt).decompose().value)

        column_values = {
            'FrameCounter': np.arange(n_records),
        }
        self.column_values = column_values

        # Timestamp is (n_frames,) in unix time
        start_utc = self.start_utc.unix
        utc = start_utc + (np.arange(n_records) * dt).decompose().value
        column_values['Timestamp'] = utc
        column_values['hwpCounts'] = self.get_hwp_column(utc)
        column_values['Flag'] = np.zeros(n_records, dtype=np.int32)
        column_values['PWV'] = self.get_pwv_column(utc)
        column_values['LOS'] = self.get_los_column(utc)
        column_values['ROLL'] = self.get_roll_column(utc)
        location = self.get_location_columns(utc)
        column_values['LON'] = location[0]
        column_values['LAT'] = location[1]

        t = Time(utc, format='unix')
        lst = t.sidereal_time('mean', location[0] * units.Unit('degree'))
        self.mjd = t.mjd
        self.lst = lst

        column_values['LST'] = lst.value
        nonsidereal = self.get_nonsidereal_columns(utc)
        column_values['NonSiderealRA'] = nonsidereal[0]
        column_values['NonSiderealDec'] = nonsidereal[1]

    @staticmethod
    def get_hwp_column(utc):
        """
        Return the hwpCounts values for the FITS data HDU.

        This needs to be updated if hwp is important.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide HWP values.

        Returns
        -------
        numpy.ndarray (int)
        """
        hwp_counts = np.zeros(utc.size, dtype=np.int32)
        return hwp_counts

    def get_pwv_column(self, utc):
        """
        Return the PWV values for the FITS data HDU.

        Linearly interpolates altitude between the first and last value.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide PWV values.

        Returns
        -------
        pwv : numpy.ndarray (float)
        """
        start = self.aircraft.start_altitude.value
        end = self.aircraft.end_altitude.value
        altitude = np.interp(utc, [utc[0], utc[-1]], [start, end]) / 1000
        pwv41k = self.info.configuration.get_float(
            'pwv41k', default=29.0)
        b = 1.0 / self.info.configuration.get_float('pwvscale', default=5.0)
        pwv = pwv41k * np.exp(-b * (altitude - 41.0))
        return pwv

    @staticmethod
    def get_los_column(utc):
        """
        Return the LOS values for the FITS data HDU.

        Linearly interpolates altitude between the first and last value.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide LOS values.

        Returns
        -------
        los : numpy.ndarray (float)
        """
        return np.zeros(utc.size, dtype=float)

    @staticmethod
    def get_roll_column(utc):
        """
        Return the ROLL values for the FITS data HDU.

        Linearly interpolates altitude between the first and last value.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide ROLL values.

        Returns
        -------
        roll : numpy.ndarray (float)
        """
        return np.zeros(utc.size, dtype=float)

    def get_location_columns(self, utc):
        """
        Return the LON/LAT values for the FITS data HDU.

        Linearly interpolates altitude between the first and last value.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide location values.

        Returns
        -------
        location : numpy.ndarray (float)
            The location of SOFIA of shape (2, n_records) with LON in
            location[0] and LAT in location[1].
        """
        location = np.zeros((2, utc.size), dtype=float)
        start_lon = self.aircraft.start_location.longitude.to('degree').value
        end_lon = self.aircraft.end_location.longitude.to('degree').value
        location[0] = np.interp(utc, [utc[0], utc[-1]], [start_lon, end_lon])
        start_lat = self.aircraft.start_location.latitude.to('degree').value
        end_lat = self.aircraft.end_location.latitude.to('degree').value
        location[1] = np.interp(utc, [utc[0], utc[-1]], [start_lat, end_lat])
        return location

    def get_nonsidereal_columns(self, utc):
        """
        Return the Nonsidereal RA/DEC values for the FITS data HDU.

        If a true nonsidereal object is to be simulated, this should
        be updated.

        Parameters
        ----------
        utc : numpy.ndarray (float)
            The UTC (unix) times for which to provide location values.

        Returns
        -------
        nonsidereal : numpy.ndarray (float)
            The nonsidereal coordinates of the object of shape (2, n_records)
            with RA (hours) in nonsidereal[0] and DEC (degree) in
            nonsidereal[1].
        """
        nonsidereal = np.zeros((2, utc.size), dtype=float)
        ra = self.source_equatorial.ra.to('hourangle').value
        dec = self.source_equatorial.dec.to('degree').value
        nonsidereal[0] = ra
        nonsidereal[1] = dec
        return nonsidereal

    def update_vpa_columns(self):
        """
        Update the VPA columns for the FITS data HDU.

        Returns
        -------
        None
        """
        n_frames = self.integration.size
        if 'TELVPA' in self.primary_header:
            self.column_values['TABS_VPA'] = np.full(
                n_frames, self.primary_header['TELVPA'])
        else:
            self.column_values['TABS_VPA'] = np.zeros(n_frames, dtype=float)

        pa = self.column_values['TABS_VPA'] * units.Unit('degree')
        self.sin_pa = np.sin(pa).value
        self.cos_pa = np.cos(pa).value
        self.column_values['SIBS_VPA'] = self.column_values['TABS_VPA'].copy()

        chop_angle = self.primary_header['CHPANGLE']
        chop_angles = chop_angle + self.column_values['TABS_VPA']
        chop_angles = Angle(chop_angles, 'degree').wrap_at(
            360 * units.Unit('degree')).to('degree').value
        self.column_values['Chop_VPA'] = chop_angles

    def update_chopper(self):
        """
        Add chopper signals to the astronomical positions.

        Currently, only CHPAMP1 is used to determine amplitude and 2-point
        chopping is simulated.  CHPNOISE in the header is used to apply
        random simulated chopper offsets up to a maximum of CHPNOISE
        arcseconds.

        Returns
        -------
        None
        """
        n_frames = self.column_values['FrameCounter'].size
        chop_r = np.zeros(n_frames, dtype='>f4')
        chop_s = chop_r.copy()

        self.column_values['sofiaChopR'] = chop_r
        self.column_values['sofiaChopS'] = chop_s

        self.chopper_position = Coordinate2D(
            np.zeros((2, n_frames)), unit='arcsec')

        if (not self.primary_header['CHOPPING']
                and 'CHPNOISE' not in self.primary_header):
            return

        # Add noise to chopper position?
        if 'CHPNOISE' in self.primary_header:
            # # A slight random walk
            # noise = (np.random.random((2, n_frames)) - 0.5) * 0.2
            # noise = np.cumsum(noise, axis=1)

            # Add normal randomness
            noise = np.random.normal(loc=0, scale=1, size=(2, n_frames))
            noise -= np.mean(noise, axis=1)[:, None]
            noise /= np.max(np.abs(noise), axis=1)[:, None]
            scale = self.primary_header['CHPNOISE'] * units.Unit('arcsec')
            noise = noise * scale
            self.chopper_position.add_x(noise[0])
            self.chopper_position.add_y(noise[1])

        dt = self.scan.info.sampling_interval.decompose().value  # Seconds
        volts_to_angle = self.scan.info.chopping.volts_to_angle
        inverted = self.scan.configuration.get_bool('chopper.invert')
        rotation = (self.column_values['Chop_VPA']
                    - self.column_values['TABS_VPA']) * units.Unit('degree')

        amplitude = self.primary_header['CHPAMP1'] * units.Unit('arcsec')
        if amplitude > 0:  # TODO: it's obviously not sinusoidal
            chop_frequency = self.primary_header['CHPFREQ']  # Hz
            t = np.arange(n_frames) * dt
            signal = np.sin(t * chop_frequency * np.pi) * amplitude
            self.chopper_position.add_x(signal)

        self.chopper_position.rotate(-rotation)
        if inverted:
            self.chopper_position.invert()

        # # Must apply reverse chopper shift
        # n = self.integration.configuration.get_int('chopper.shift',
        #                                            default=0)
        # if n != 0:
        #     self.chopper_position.shift(-n, fill_value=0.0)

        voltages = self.chopper_position.coordinates / -volts_to_angle
        chop_s[...] = voltages[0].value
        chop_r[...] = voltages[1].value

    def update_astronomical_columns(self):
        """
        Update the astronomical columns for the FITS data HDU.

        Returns
        -------
        None
        """
        scan_pattern = self.primary_header['SCNPATT'].strip().lower()
        if scan_pattern == 'daisy':
            equatorial = self.get_daisy_equatorial()
        elif scan_pattern == 'lissajous':
            equatorial = self.get_lissajous_equatorial()
        else:
            raise ValueError(f"Scan pattern {scan_pattern} not implemented.")

        n_frames = self.integration.size
        if equatorial.size > n_frames:
            equatorial = equatorial[:n_frames]
        elif equatorial.size < n_frames:
            coordinates = np.empty((2, n_frames), dtype=float)
            coordinates[:, :equatorial.size] = equatorial.coordinates.value
            last = equatorial[-1].coordinates.value
            coordinates[:, equatorial.size:] = last[:, None]
            equatorial = EquatorialCoordinates(
                coordinates, unit=equatorial.unit)

        lst = self.column_values['LST'] * units.Unit('hour')
        lon = self.column_values['LON'] * units.Unit('degree')
        lat = self.column_values['LAT'] * units.Unit('degree')
        site = GeodeticCoordinates([lon, lat])

        horizontal_offset = equatorial.get_native_offset_from(
            self.source_equatorial)
        x = horizontal_offset.x.copy()
        y = horizontal_offset.y.copy()
        horizontal_offset.set_x((self.cos_pa * x) + (self.sin_pa * y))
        horizontal_offset.set_y((self.cos_pa * y) - (self.sin_pa * x))

        apparent_equatorial = equatorial.copy()
        self.info.astrometry.to_apparent.precess(apparent_equatorial)
        horizontal = apparent_equatorial.to_horizontal(site, lst)

        self.horizontal_corrected = horizontal.copy()
        self.equatorial_corrected = equatorial.copy()
        self.horizontal_offset_corrected = horizontal_offset.copy()
        self.apparent_equatorial_corrected = apparent_equatorial.copy()

        # Must apply chopper shift

        n = self.integration.configuration.get_int('chopper.shift', default=0)
        if n != 0:
            self.chopper_position.shift(n, fill_value=0.0)

        # Apply the chopper position
        horizontal_offset.subtract(self.chopper_position)
        horizontal.subtract_offset(self.chopper_position)

        equatorial_offset = self.chopper_position.copy()
        x = equatorial_offset.x.copy()
        y = equatorial_offset.y.copy()
        equatorial_offset.set_x((self.cos_pa * x) - (self.sin_pa * y))
        equatorial_offset.set_y((self.sin_pa * x) + (self.cos_pa * y))
        equatorial.subtract_native_offset(equatorial_offset)

        apparent_equatorial = equatorial.copy()
        self.info.astrometry.to_apparent.precess(apparent_equatorial)
        self.apparent_equatorial = apparent_equatorial

        # Must redo horizontal
        horizontal = apparent_equatorial.to_horizontal(site, lst)

        self.site = site
        self.horizontal = horizontal
        self.equatorial = equatorial
        self.horizontal_offset = horizontal_offset

        self.column_values['RA'] = equatorial.ra.to('hourangle').value
        self.column_values['DEC'] = equatorial.dec.to('degree').value
        self.column_values['AZ'] = horizontal.az.to('degree').value
        self.column_values['EL'] = horizontal.el.to('degree').value

    def get_daisy_equatorial(self):
        """
        Return the equatorial offsets for a daisy scan pattern.

        Returns
        -------
        equatorial_offsets : EquatorialCoordinates
        """
        n_oscillations = self.primary_header['SCNDNOSC']
        radius = self.primary_header['SCNDRAD'] * units.Unit('arcsec')
        radial_period = self.primary_header['SCNDPER'] * units.Unit('second')
        return daisy_pattern_equatorial(self.source_equatorial,
                                        self.scan.info.sampling_interval,
                                        n_oscillations=n_oscillations,
                                        radius=radius,
                                        radial_period=radial_period)

    def get_lissajous_equatorial(self):
        """
        Return the equatorial offsets for a Lissajous scan pattern.

        Returns
        -------
        equatorial_offsets : EquatorialCoordinates
        """
        h = self.primary_header
        width = h['SCNAMPXL'] * units.Unit('arcsec')
        height = h['SCNAMPEL'] * units.Unit('arcsec')
        ratio = h['SCNFQRAT']
        delta = h['SCNPHASE'] * units.Unit('degree')

        scan_time = h['EXPTIME'] * units.Unit('second')
        scan_rate = h['SCNRATE'] * units.Unit('arcsec/second')
        n_oscillations = h.get('SCNNOSC', 20)

        oscillation_period = scan_time / n_oscillations
        scan_time = n_oscillations * oscillation_period

        # Calculate the distance travelled for a single oscillation.
        r = np.hypot(width / 2, height / 2) * 2 * np.pi
        oscillation_period = r / scan_rate
        n_oscillations = (scan_time / oscillation_period).decompose().value

        equatorial = lissajous_pattern_equatorial(
            self.source_equatorial,
            self.scan.info.sampling_interval,
            width=width, height=height, delta=delta, ratio=ratio,
            n_oscillations=n_oscillations,
            oscillation_period=oscillation_period)

        return equatorial

    def get_data_hdu(self):
        """
        Return a FITS BinTable containing current data.

        Returns
        -------
        fits.BinTableHDU
        """
        fits_cols = self.info.detector_array.FITS_COLS
        fits_rows = self.info.detector_array.FITS_ROWS

        n = self.integration.size
        data_shape = (n, fits_rows, fits_cols)

        min_points = fits_cols * fits_rows
        if n < min_points:
            n_array = n * fits_rows
        else:
            n_array = n

        dim = f'({fits_cols}, {fits_rows})'
        cols = []

        for name, coldef in self.data_column_definitions.items():
            if name in ['SQ1Feedback', 'FluxJumps']:
                continue
            value = self.column_values.get(name)
            if value is None:
                continue
            unit, fmt = coldef
            column = fits.Column(name=name, unit=unit, format=fmt, array=value)
            cols.append(column)

        key = 'SQ1Feedback'
        values = self.column_values.get(key,
                                        np.ones(data_shape, dtype='int32'))
        cols.append(fits.Column(
            name=key, format=f'{n_array}J', dim=dim, array=values))

        key = 'FluxJumps'
        values = self.column_values.get(key, np.zeros(data_shape, dtype='>i2'))
        cols.append(fits.Column(
            name=key, format=f'{n_array}I', dim=dim, array=values))

        hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        hdu.header['EXTNAME'] = 'Timestream'
        return hdu

    def create_source_model(self):
        """
        Create the simulated source model.

        Returns
        -------
        None
        """
        source_type = self.primary_header.get('SRCTYPE', 'point_source')
        source_type = source_type.strip().lower()
        if source_type in ['point_source', 'extended']:
            model_name = 'single_gaussian'
            fwhm = self.primary_header.get('SRCSIZE') * units.Unit('arcsec')
            self.source_model = SimulatedSource.get_source_model(
                model_name, fwhm=fwhm)
        else:
            raise ValueError(
                f"{source_type} simulated source is not implemented.")

    def create_simulated_data(self):
        """
        Populate the SQ1Feedback column values for the data HDU.

        Returns
        -------
        None
        """
        integration = self.integration
        frames = integration.frames
        channels = integration.channels
        configuration = integration.configuration

        frames.equatorial = self.equatorial.copy()
        frames.horizontal = self.horizontal.copy()
        frames.horizontal_offset = self.horizontal_offset.copy()
        frames.chopper_position = self.chopper_position.copy()
        frames.validate()

        self.projection = SphericalProjection.for_name(
            configuration.get_string('projection', default='gnomonic'))
        self.projection.set_reference(self.source_equatorial)
        self.projector = AstroProjector(self.projection)
        offsets = frames.project(channels.data.position, self.projector)
        offsets.change_unit('arcsec')
        self.model_offsets = offsets

        # Source peak is 1.
        source_data = self.source_model(offsets)
        if isinstance(source_data, units.Quantity):
            source_data = source_data.value

        # Multiply values by this
        instrument_gain = 1 / integration.get_default_scaling_factor()

        abs_gain = np.abs(instrument_gain)

        t = self.primary_header['EXPTIME']
        g2v = channels.data.gain ** 2
        v = channels.data.variance
        nzi = v != 0
        g2v[nzi] /= v[nzi]
        g2v[~nzi] = np.nan
        nan = np.full(g2v.size, True)
        mapping = channels.get_mapping_pixels()
        nan[mapping.indices] = False
        g2v[nan] = np.nan
        nefd = np.sqrt(mapping.size * t * np.nansum(g2v)) / abs_gain
        scale = nefd * self.primary_header.get('SRCAMP', 1.0)
        self.source_max = scale
        source_data *= scale

        source_valid = np.isfinite(source_data)

        subarray_norm = channels.subarray_gain_renorm[channels.data.sub]
        source_data *= subarray_norm[None]

        if 'SRCS2N' in self.primary_header:
            s2n = float(self.primary_header['SRCS2N'])
            noise = np.random.normal(loc=0, scale=self.source_max / s2n,
                                     size=source_data.shape)
            source_data += noise

        channels.load_temporary_hardware_gains()
        inv_gains = channels.data.temp.copy()
        nzi = inv_gains != 0
        inv_gains[nzi] = 1 / inv_gains[nzi]
        int_nf.detector_stage(frame_data=source_data,
                              frame_valid=np.full(frames.size, True),
                              channel_indices=np.arange(inv_gains.size),
                              channel_hardware_gain=inv_gains)

        # Add a DC offset
        offset = np.nanmin(source_data) - 1.0
        source_data -= offset
        source_data[~source_valid] = 0.0

        self.source_data = source_data

        fits_cols = self.info.detector_array.FITS_COLS
        fits_rows = self.info.detector_array.FITS_ROWS
        n = integration.size
        data_shape = (n, fits_rows, fits_cols)
        data = np.zeros(data_shape, dtype='>i4')
        data[:, channels.data.fits_row, channels.data.fits_col] = source_data
        self.column_values['SQ1Feedback'] = data

    def create_simulated_jumps(self):
        """
        Add jumps to the data HDU columns.

        Returns
        -------
        None
        """
        fits_cols = self.info.detector_array.FITS_COLS
        fits_rows = self.info.detector_array.FITS_ROWS
        n = self.integration.size
        shape = (n, fits_rows, fits_cols)
        jump_array = np.zeros(shape, dtype='>i2')
        self.column_values['FluxJumps'] = jump_array

        if 'JUMPCHAN' not in self.primary_header:
            return

        channels = self.integration.channels
        jump_channels = self.primary_header['JUMPCHAN']
        if jump_channels.lower() == 'all':
            channel_indices = np.arange(channels.size)
        else:
            fixed_indices = get_int_list(self.primary_header['JUMPCHAN'])
            channel_indices = channels.data.find_fixed_indices(fixed_indices)

        if channel_indices.size == 0:
            return

        if 'JUMPFRMS' not in self.primary_header:
            frame_indices = np.asarray([n // 2])  # One in the middle
        else:
            frame_indices = get_int_list(self.primary_header['JUMPFRMS'])

        if frame_indices.size == 0:
            return

        jumps = np.zeros((n, channels.size), dtype=int)
        start_jumps = jumps[0]
        start_jumps[channel_indices] = np.arange(channel_indices.size) + 1
        jumps[...] = start_jumps[None, :]

        jump_size = self.primary_header.get('JUMPSIZE', 1)
        jump_correction = jump_size * channels.data.jump[channel_indices]
        data = self.column_values['SQ1Feedback']
        jump_correction = np.round(jump_correction).astype(data.dtype)

        fits_col, fits_row = channels.data.fits_col, channels.data.fits_row
        col_i = fits_col[channel_indices]
        row_i = fits_row[channel_indices]

        for jump_frame in frame_indices:
            jumps[jump_frame:, channel_indices] += jump_size
            data[jump_frame:, row_i, col_i] += jump_correction

        jump_array[:, row_i, col_i] = jumps[:, channel_indices]

# Keys not provided yet...
# DBRA0   = '1h16m21.670s'       / Drift 0 begin RA [hours]
# DBDEC0  = '7d24m24.043s'       / Drift 0 begin Dec [deg]
# DBTIME0 =      1.48169779436E9 / Drift 0 begin time [seconds]
# DARA0   = '1h16m21.669s'       / Drift 0 after RA [hours]
# DADEC0  = '7d24m23.757s'       / Drift 0 after Dec [deg]
# DATIME0 =      1.48169779711E9 / Drift 0 after time [seconds]

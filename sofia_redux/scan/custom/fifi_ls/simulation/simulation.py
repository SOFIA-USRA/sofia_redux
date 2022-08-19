from abc import ABC
from astropy import units
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.time import Time
import getpass
import numpy as np
import warnings

from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
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
from sofia_redux.scan.utilities.utils import (
    get_hms_time, get_dms_angle)


__all__ = ['FifiLsSimulation']


class FifiLsSimulation(ABC):

    # FITS header keys specific to this simulation or used to override defaults
    sim_keys = {'CHPNOISE', 'SCNCONST', 'SCNDRAD', 'SCNDPER', 'SCNDNOSC',
                'SRCTYPE', 'SRCSIZE', 'SRCAMP', 'SRCAMP', 'SRCS2N',
                'OBSRA', 'OBSDEC', 'DATE-OBS', 'LON_STA', 'LAT_STA'}

    default_values = {
        'SIMPLE': True,
        'DATASRC': 'ASTRO',
        'KWDICT': 'DCS_SI_01_F',
        'AOT_ID': '',
        'FILEGP_R': 'R1.1',
        'FILEGP_B': 'B1.1',
        'OBSTYPE': 'SKY',
        'PROCSTAT': 'LEVEL_3',
        'HEADSTAT': 'SIMULATED',
        'DEPLOY': 'DAOF',
        'FLIGHTLG': '9',
        'ORIGIN': 'FIFI-LS',
        'OBSERVER': getpass.getuser(),
        'CREATOR': 'SOFSCAN FIFI-LS Simulator',
        'FILTVERS': '2143L2/1449R01/2145L01/2144B02A/1448B02B 12/01/2017',
        'OPERATOR': getpass.getuser(),
        'FILENAME': 'simulated_fifi_ls.fits',
        'FIFISTRT': 2947840,
        'FIFIEND': 2973440,
        'WVZ_STA': 56.695,
        'WVZ_END': 59.222,
        'TEMP_OUT': -49.5,
        'TEMPPRI1': -24.85,
        'TEMPPRI2': -24.05,
        'TEMPPRI3': -24.95,
        'TEMPSEC1': -22.05,
        'ALTI_STA': 41032.0,
        'ALTI_END': 41017.0,
        'AIRSPEED': 498.312,
        'GRDSPEED': 588.531,
        'LAT_STA': 31.8846,
        'LON_STA': -108.37,
        'LAT_END': 31.8835,
        'LON_END': -108.201,
        'HEADING': 90.1298,
        'TRACKANG': 90.4244,
        'TELESCOP': 'SOFIA 2.5m',
        'TELCONF': 'DEFAULT',
        'TELEQUI': 'j2000',
        'FOCUS_ST': 650.458,
        'FOCUS_EN': 650.46,
        'TSC-STAT': 'STAB_INERTIAL_ONGOING',
        'FBC-STAT': 'FBC_ON',
        'EQUINOX': 2000.0,
        'TRACMODE': 'OFFSET',
        'TRACERR': True,
        'CHOPPING': True,
        'NODDING': True,
        'DITHERING': True,
        'INSTRUME': 'FIFI-LS',
        'DATATYPE': 'OTHER',
        'INSTCFG': 'DUAL_CHANNEL',
        'INSTMODE': 'OTF_TP',
        'MCCSMODE': 'fifi-ls_standard',
        'SPECTEL1': 'FIF_BLUE',
        'SPECTEL2': 'NONE',
        'DETCHAN': 'BLUE',
        'RESTWAV': 51.815,
        'CHPFREQ': 1.0,
        'CHPPROF': '2-POINT',
        'CHPSYM': 'no_chop',
        'CHPAMP1': 0.0,
        'CHPAMP2': 0.0,
        'CHPCRSYS': 'tarf',
        'CHPANGLE': 90.0,
        'CHPTIP': 0.0,
        'CHPTILT': 0.0,
        'CHPPHASE': 0,
        'NODTIME': 51.2,
        'NODN': 0,
        'NODSETL': 0.0,
        'NODAMP': 60.0,
        'NODBEAM': 'UNKNOWN',
        'NODPATT': 'ABBA',
        'NODSTYLE': 'C2NC2',
        'NODCRSYS': 'tarf',
        'NODANGLE': 90.0,
        'SKYSPEED': 4.0,
        'VELANGLE': 90.0,
        'TRK_DRTN': 30.0,
        'OBSLAMV': 4.0,
        'OBSBETV': -8.2041371e-10,
        'ALPHA': 0.004000371,
        'START': 1582603244.09346,
        'OTFSTART': 1582615040.72062,
        'UNIXSTRT': 1582615036.547109,
        'STATICAIRPRESS': 5.257,
        'BAROALTITUDE': 41031.0,
        'HOURANGLE': 0.418476,
        'SKYLOSRATE': -36.5826,
        'OBSCOORDEQUINOX': 2000.0,
        'PLATSCAL': 4.2304565,
        'TEL_ANGL': 175.85293,
        'TELONTRK': 'N',
        'TELOSTTK': 'N',
        'CHOPOK': 'N',
        'CHOPERR': 0,
        'DATAPROD': 'RAW',
        'COORDSYS': 'J2000',
        'CRDSYSMP': 'J2000',
        'CRDSYSOF': 'J2000',
        'PRIMARAY': 'BLUE',
        'DICHROIC': 130,
        'G_ORD_B': 2,
        'G_FLT_B': 2,
        'G_WAVE_B': 51.853,
        'G_CYC_B': 1,
        'G_STRT_B': 735588,
        'G_PSUP_B': 1,
        'G_SZUP_B': 400,
        'G_PSDN_B': 0,
        'G_SZDN_B': 0,
        'G_WAVE_R': 157.857,
        'G_CYC_R': 1,
        'G_STRT_R': 735588,
        'G_PSUP_R': 1,
        'G_SZUP_R': 365,
        'G_PSDN_R': 0,
        'G_SZDN_R': 0,
        'RAMPLN_B': 32,
        'RAMPLN_R': 32,
        'C_SCHEME': '2POINT',
        'C_CRDSYS': 'TARF',
        'C_AMP': 0.0,
        'C_TIP': 0.0,
        'C_BEAM': 1,
        'C_POSANG': 0.0,
        'C_CYC_B': 200,
        'C_CYC_R': 200,
        'C_PHASE': 0.0,
        'C_CHOPLN': 64,
        'CAP_R': 1330,
        'CAP_B': 1330,
        'ZBIAS_B': 75.0,
        'BIASR_B': 0.0,
        'HEATER_B': 0.0,
        'ZBIAS_R': 60.0,
        'BIASR_R': 0.0,
        'HEATER_R': 0.0,
        'CALSTMP': 200.0,
        'WAVECENT': 51.853,
        'FILEGPID': 'W40_63.184',
        'TMPCALSC': 39.568,
        'TMPOPTBN': 6.36,
        'TMPDETS': 1.66,
        'GROUP': 'raw_simulated.fits',
        'ATRNFILE': 'atran_41K_40deg_40-300mum.fits',
        'BDPXFILE': 'BADPIXELS_202002_B.TXT',
        'BGLEVL_A': 149536689451054.7,
        'BGLEVL_B': 149555363218331.0,
        'DATAQUAL': 'UNKNOWN',
        'DETECTOR': 'UNKNOWN',
        'DETSIZE': 'UNKNOWN',
        'DITHER': False,
        'DTHCRSYS': 'UNKNOWN',
        'DTHINDEX': -9999,
        'DTHNPOS': -9999,
        'DTHOFFS': -9999.0,
        'DTHPATT': 'UNKNOWN',
        'DTHXOFF': -9999.0,
        'DTHYOFF': -9999.0,
        'FILENUM': '00353-00354',
        'FILEREV': 'UNKNOWN',
        'FLATFILE': 'flat_files/spatialFlatB2.txt[20210401],flat_files/spectralFlatsB2D130.fits',
        'IMAGEID': -9999,
        'MAPCRSYS': 'UNKNOWN',
        'MAPINTX': -9999.0,
        'MAPINTY': -9999.0,
        'MAPNXPOS': -9999,
        'MAPNYPOS': -9999,
        'MAPPING': False,
        'NEXP': 1,
        'PIPELINE': 'FIFI_LS_REDUX',
        'PIPEVERS': '2_6_1_dev2+g9ea2462',
        'PIXSCAL': -9999.0,
        'PLANID': 'UNKNOWN',
        'PRODTYPE': 'wavelength_shifted',
        'RAWUNITS': 'adu/(Hz s)',
        'RESFILE': 'spectral_resolution.txt',
        'RESOLUN': -9999.0,
        'RSPNFILE': 'Response_Blue_D130_Ord2_20190705v2.fits',
        'SCANNING': False,
        'SCNDEC0': -9999.0,
        'SCNDECF': -9999.0,
        'SCNDIR': -9999.0,
        'SCNRA0': -9999.0,
        'SCNRAF': -9999.0,
        'SCNRATE': -9999.0,
        'SIBS_X': -9999,
        'SIBS_Y': -9999,
        'SLIT': 'UNKNOWN',
        'SPATFILE': 'pixel_pos_blue_20150828.txt',
        'SUBARRNO': -9999,
        'TELAPSE': 47.0,
        'WAVEFILE': 'FIFI_LS_WaveCal_Coeffs.txt',
        'WVSCALE': -9999.0,
        'XPOSURE': 44.927418272,
        'RADESYS': 'FK5',
        'TIMESYS': 'UTC',
        'TIMEUNIT': 's',
        'CHANNEL': 'BLUE',
        'NGRATING': 1,
        'CHOPNUM': 0,
        'BUNIT': 'Jy/pixel',
        'CALERR': 0.05933010658089425,
        'BARYSHFT': -3.6425523026739e-05,
        'LSRSHFT': 2.06475280148922e-05,
        'SRCAMP': 100.0,
        'SRCS2N': 20.0}

    default_comments = {
        'SIMPLE': 'conforms to FITS standard',
        'BITPIX': 'array data type',
        'NAXIS': 'number of array dimensions',
        'DATASRC': 'Data source',
        'SRCTYPE': 'Source type',
        'KWDICT': 'SOFIA Keyword dictionary version',
        'OBS_ID': 'SOFIA observation identification',
        'OBJECT': 'Object name',
        'AOT_ID': 'Astronomical observation template ID',
        'FILEGP_R': 'Red file group for pipeline',
        'FILEGP_B': 'Blue file group for pipeline',
        'OBSTYPE': 'Observation type',
        'PROCSTAT': 'Processing status',
        'HEADSTAT': 'Header status',
        'DEPLOY': 'Site deployment',
        'MISSN-ID': 'Mission ID',
        'FLIGHTLG': 'Flight leg',
        'ORIGIN': 'Origin of FITS file',
        'OBSERVER': 'Observer(s)',
        'CREATOR': 'File creation task',
        'FILTVERS': 'filter set des',
        'OPERATOR': 'Telescope operator',
        'FILENAME': 'Name of host file',
        'DATE': 'Date of file creation',
        'FIFISTRT': 'Start FIFI time',
        'FIFIEND': 'End FIFI time',
        'DATE-OBS': 'UTC Date of exposure start',
        'UTCSTART': 'UTC of exposure start',
        'UTCEND': 'UTC of exposure end',
        'WVZ_STA': 'Water vapor at zenith, obs. start',
        'WVZ_END': 'Water vapor at zenith, obs. end',
        'TEMP_OUT': 'Air temperature outside aircraft',
        'TEMPPRI1': 'Temperature of primary mirror',
        'TEMPPRI2': 'Temperature of primary mirror',
        'TEMPPRI3': 'Temperature of primary mirror',
        'TEMPSEC1': 'Temperature of secondary mirror',
        'ALTI_STA': 'Altitude at start of observation',
        'ALTI_END': 'Altitude at end of observation',
        'AIRSPEED': 'True aircraft speed',
        'GRDSPEED': 'Aircraft ground speed',
        'LAT_STA': 'Aircraft latitude, start of obs',
        'LON_STA': 'Aircraft longitude, start of obs',
        'LAT_END': 'Aircraft latitude, end of obs.',
        'LON_END': 'Aircraft longitude, end of obs.',
        'HEADING': 'Aircraft true heading',
        'TRACKANG': 'Aircraft track angle',
        'TELESCOP': 'Telescope name',
        'TELCONF': 'Telescope configuration',
        'TELRA': 'SI boresight RA',
        'TELDEC': 'SI boresight Dec',
        'TELVPA': 'SI boresight VPA',
        'TELEQUI': 'Equinox of ERF coordinates',
        'LASTREW': 'Time of last rewind (UTC)',
        'FOCUS_ST': 'Telescope focus (um), obs.start',
        'FOCUS_EN': 'Telescope focus (um), obs.end',
        'TELEL': 'Telescope elevation',
        'TELXEL': 'Telescope cross elevation',
        'TELLOS': 'Telescope LOS',
        'TSC-STAT': 'TASCU status at obs. end',
        'FBC-STAT': 'FBC status at obs. end',
        'OBSRA': 'RA - requested',
        'OBSDEC': 'Dec - requested',
        'EQUINOX': 'Equinox of celestial CS',
        'ZA_START': 'Telescope zenith angle, obs. start',
        'ZA_END': 'Telescope zenith angle, obs. end',
        'TRACMODE': 'Tracking mode',
        'TRACERR': 'Tracking error flag',
        'CHOPPING': 'Chopping flag',
        'NODDING': 'Nodding flag',
        'DITHERING': '',
        'INSTRUME': 'Instrument name',
        'DATATYPE': 'Data type',
        'INSTCFG': 'Instrument configuration',
        'INSTMODE': 'Instrument observing mode',
        'MCCSMODE': 'MCCS SI Mode',
        'EXPTIME': 'Total on-source exposure time',
        'SPECTEL1': 'First spectral element',
        'SPECTEL2': 'Second spectral element',
        'DETCHAN': 'Detector channel',
        'RESTWAV': 'Wavelength before shift correction',
        'CHPFREQ': 'Chop frequency (Hz)',
        'CHPPROF': 'Chop profile; 2 or 3 point',
        'CHPSYM': 'Chop symmetry',
        'CHPAMP1': 'Chop amplitude 1 (arcsec)',
        'CHPAMP2': 'Chop amplitude 2 (arcsec)',
        'CHPCRSYS': 'Chop coordinate system',
        'CHPANGLE': 'Chop angle',
        'CHPTIP': 'Chop tip',
        'CHPTILT': 'Chop tilt',
        'CHPPHASE': 'Chop phase (milliseconds)',
        'NODTIME': 'Nod time',
        'NODN': 'Nod cycles',
        'NODSETL': 'Nod settle time',
        'NODAMP': 'Nod amplitude on sky',
        'NODBEAM': 'Nod beam position',
        'NODPATT': 'Nodding pattern, one cycle',
        'NODSTYLE': 'Chop/nod style',
        'NODCRSYS': 'Coordinate system for nod angle',
        'NODANGLE': 'Nod angle',
        'SKYSPEED': 'Module of the velocity vector for OTF scan',
        'VELANGLE': 'Angle of the velocity vector for OTF scan',
        'TRK_DRTN': 'Duration of the OTF scan',
        'OBSLAMV': 'Projection of the velocity along Lambda',
        'OBSBETV': 'Projection of the velocity along Beta',
        'ALPHA': 'alpha value in fifiTime correlation',
        'START': 'start value in fifiTime correlation',
        'OTFSTART': 'time when scan is commmanded to start',
        'UNIXSTRT': 'start in unix time fifiTime correlation',
        'ALTITUDE': 'GPS MSL altitude [feet]',
        'STATICAIRPRESS': 'Static outside air pressure [inch Hg]',
        'BAROALTITUDE': 'Barometric altitude [feet]',
        'HOURANGLE': 'Telescope Hour Angle (used by los_monitor to ca',
        'SKYLOSRATE': 'Sky rotation rate estimated from actual conditi',
        'OBSCOORDEQUINOX': 'Coordinate Equinox for Requested Right Ascens',
        'PLATSCAL': 'Plate scale of focal plane (arcsec/mm)',
        'TEL_ANGL': 'ccw angle from first coordinate of focal plane',
        'TELONTRK': 'kosma tel_on_track state',
        'TELOSTTK': 'kosma tel_lost_track state',
        'CHOPOK': 'kosma chop_ok state',
        'CHOPERR': 'kosma chop_error value (OK=0)',
        'DATAPROD': 'Type of FIFI-LS data product',
        'OBJ_NAME': 'Name of astronomical object observed',
        'COORDSYS': 'Obs Coordinate System',
        'OBSLAM': 'First angle in deg',
        'OBSBET': 'Second angle in deg',
        'CRDSYSMP': 'Mapping Coordinate System',
        'DLAM_MAP': 'Map offset (arcsec)',
        'DBET_MAP': 'Map offset (arcsec)',
        'CRDSYSOF': 'Coordinate System for Off pos.',
        'DLAM_OFF': 'Off offset in arcsec',
        'DBET_OFF': 'Off offset in arcsec',
        'DET_ANGL': 'Detector y-axis EofN in deg',
        'PRIMARAY': 'Primary Array',
        'DICHROIC': 'Dichroic wavelength (um)',
        'G_ORD_B': 'Blue grating order to be used',
        'G_FLT_B': 'Blue grating order filter to be used',
        'G_WAVE_B': 'Wavelength to be observed in um INFO ONLY',
        'G_CYC_B': 'Number of grating cycles (up-down)',
        'G_STRT_B': 'Absolute starting value (inductosyn)',
        'G_PSUP_B': 'Number of grating position up in 1 cycle',
        'G_SZUP_B': 'Step size on the way up (inductosyn)',
        'G_PSDN_B': 'Number of grating position down in 1 cycle',
        'G_SZDN_B': 'Step size on the way down (inductosyn)',
        'G_WAVE_R': 'Wavelength to be observed in um INFO ONLY',
        'G_CYC_R': 'Number of grating cycles (up-down)',
        'G_STRT_R': 'Absolute starting value (inductosyn)',
        'G_PSUP_R': 'Number of grating position up in 1 cycle',
        'G_SZUP_R': 'Step size on the way up (inductosyn)',
        'G_PSDN_R': 'Number of grating position down in 1 cycle',
        'G_SZDN_R': 'Step size on the way down (inductosyn)',
        'RAMPLN_B': 'Number of readouts per ramp',
        'RAMPLN_R': 'Number of readouts per ramp',
        'C_SCHEME': 'Chopper scheme; 2POINT or 4POINT',
        'C_CRDSYS': 'Coordinate System for chopping',
        'C_AMP': 'chop amp in arcsec',
        'C_TIP': 'relative chop tip',
        'C_BEAM': 'nod phase',
        'C_POSANG': 'deg',
        'C_CYC_B': 'chopping cycles per grating position',
        'C_CYC_R': 'chopping cycles per grating position',
        'C_PHASE': 'phase shift of chopper signal relative to R/O i',
        'C_CHOPLN': 'Number of readouts per chop position',
        'CAP_R': 'Integrating capacitors in pF',
        'CAP_B': 'Integrating capacitors in pF',
        'ZBIAS_B': 'Voltage in mV',
        'BIASR_B': 'Voltage in mV',
        'HEATER_B': 'Voltage in mV',
        'ZBIAS_R': 'Voltage in mV',
        'BIASR_R': 'Voltage in mV',
        'HEATER_R': 'Voltage in mV',
        'CALSTMP': 'Cal src temp in K',
        'WAVECENT': 'Central wavelength of observation',
        'FILEGPID': 'File group for pipeline',
        'TMPCALSC': 'As read temp of cal source in K',
        'TMPOPTBN': 'As read temp of optical bench in K',
        'TMPDETS': 'As read temp of detectors in K',
        'GROUP': '',
        'AOR_ID': 'Astronomical observation request ID',
        'ASSC_AOR': 'All input AOR-IDs',
        'ASSC_MSN': 'All input MISSN-IDs',
        'ASSC_OBS': 'All input OBS-ID',
        'ATRNFILE': 'ATRAN file name',
        'BDPXFILE': 'Bad pixel file name',
        'BGLEVL_A': 'Background level A nod (Jy/pixel)',
        'BGLEVL_B': 'Background level B nod (Jy/pixel)',
        'DATAQUAL': 'Data quality assessment',
        'DATE-BEG': 'UTC Date of exposure start',
        'DATE-END': 'UTC Date of exposure end',
        'DETECTOR': 'Detector name',
        'DETSIZE': 'Detector size',
        'DITHER': 'Dithering flag',
        'DTHCRSYS': 'Dither coordinate system',
        'DTHINDEX': 'Dither position index',
        'DTHNPOS': 'Number of dither positions',
        'DTHOFFS': 'Dither offset (arcsec)',
        'DTHPATT': 'Dither pattern',
        'DTHXOFF': 'Dither offset in X axis (arcsec)',
        'DTHYOFF': 'Dither offset in Y axis (arcsec)',
        'FILENUM': 'Raw file number',
        'FILEREV': 'File revision identifier',
        'FLATFILE': 'Flat filename',
        'IMAGEID': 'Image identification index',
        'MAPCRSYS': 'Coordinate system for mapping',
        'MAPINTX': 'Mapping step interval in x (arcmin)',
        'MAPINTY': 'Mapping step interval in y (arcmin)',
        'MAPNXPOS': 'Number of map positions in x',
        'MAPNYPOS': 'Number of map positions in y',
        'MAPPING': 'Mapping flag',
        'NEXP': 'Number of exposures in source',
        'PIPELINE': 'Pipeline/processing software',
        'PIPEVERS': 'Pipeline version',
        'PIXSCAL': 'Pixel scale',
        'PLANID': 'Observing plan identification',
        'PRODTYPE': 'Product type',
        'RAWUNITS': 'Raw data units before calibration',
        'RESFILE': 'Spectral resolution file name',
        'RESOLUN': 'Spectral resolution of observation',
        'RSPNFILE': 'Instrumental response file',
        'SCANNING': 'Scanning flag',
        'SCNDEC0': 'Start of scan - Dec.',
        'SCNDECF': 'End of scan - Dec.',
        'SCNDIR': 'Scan direction',
        'SCNRA0': 'Start of scan - RA',
        'SCNRAF': 'End of scan - Dec.',
        'SCNRATE': 'Scan rate',
        'SIBS_X': 'SI Boresight (x)',
        'SIBS_Y': 'SI Boresight (y)',
        'SLIT': 'Instrument slit',
        'SPATFILE': 'Spatial calibration file',
        'SUBARRNO': 'Number of subarrays used',
        'TELAPSE': 'Time elapsed',
        'WAVEFILE': 'Wavelength calibration file',
        'WVSCALE': 'Transmission scaling factor, given WV',
        'XPOSURE': 'Exposure time [s]',
        'RADESYS': 'Celestial CS convention',
        'TIMESYS': 'Time system',
        'TIMEUNIT': 'Time unit',
        'CHANNEL': 'Detector channel',
        'NGRATING': 'Number of grating positions',
        'CHOPNUM': 'Chop number',
        'SKY_ANGL': 'Sky angle after calibration (deg)',
        'BUNIT': 'Data units',
        'CALERR': 'Overall fractional flux cal error',
        'BARYSHFT': 'Barycentric motion dl/l shift (applied)',
        'LSRSHFT': 'Additional dl/l shift to LSR (unapplied)',
        'SCNAMPEL': 'Lissajous scan amplitude in elevation [arcsec]',
        'SCNAMPXL': 'Lissajous scan amplitude in cross-elevation [arcsec]',
        'SCNDUR': 'Lissajous requested scan duration [sec]',
        'SCNFQRAT': 'Lissajous pattern frequency ratio',
        'SCNPHASE': 'Lissajous pattern relative phase offset [deg]',
        'SCNTOFF': 'Lissajous pattern relative time offset [sec]',
        'SCNCONST': 'Scanned at constant speed',
        'SCNDRAD': 'Daisy scan radius [arcsec]',
        'SCNDPER': 'Daisy scan radial period [seconds]',
        'SCNDNOSC': 'Daisy scan number of oscillations',
        'SRCAMP': 'The simulated source amplitude (Jy/pixel)',
        'SRCS2N': 'The simulated source S2N'
    }

    def __init__(self, info=None):
        """
        Initialize a FIFI-LS simulation.

        Parameters
        ----------
        info : FifiLsInfo
        """
        if info is None:
            info = FifiLsInfo()
            info.read_configuration()

        if not isinstance(info, FifiLsInfo):
            raise ValueError(
                f"Simulation must be initialized with {FifiLsInfo}")
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
        self.horizontal_offset = None
        self.lst = None
        self.mjd = None
        self.site = None
        self.sin_pa = None
        self.cos_pa = None
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
              header : The primary header
          - hdul[1] = ImageHDU
              data : FLUX values (n_frames, n_spexels, n_spaxels)
          - hdul[2] = ImageHDU
              data : STDDEV values (n_frames, n_spexels, n_spaxels)
          - hdul[3] = ImageHDU
              data : UNCORRECTED_FLUX (n_frames, n_spexels, n_spaxels)
          - hdul[4] = ImageHDU
              data : UNCORRECTED_STDDEV (n_frames, n_spexels, n_spaxels)
          - hdul[5] = ImageHDU
              data : LAMBDA (n_spexels, n_spaxels)
          - hdul[6] = ImageHDU
              data : UNCORRECTED_LAMBDA (n_spexels, n_spaxels)
          - hdul[7] = ImageHDU
              data : XS (n_frames, n_spexels, n_spaxels)
          - hdul[8] = ImageHDU
              data : YS (n_frames, n_spexels, n_spaxels)
          - hdul[9] = ImageHDU
              data : RA (n_frames, n_spexels, n_spaxels)
          - hdul[10] = ImageHDU
              data : DEC (n_frames, n_spexels, n_spaxels)
          - hdul[11] = ImageHDU
              data : ATRAN (n_spexels, n_spaxels)
          - hdul[12] = ImageHDU
              data : RESPONSE (n_spexels, n_spaxels)
          - hdul[13] = ImageHDU
              data : UNSMOOTHED_ATRAN (2, n_atran_values)

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
        self.create_lambda_hdu()
        self.create_atran_response_hdu()
        self.generate_scan_pattern()
        self.create_simulated_data()

        # Reorder the HDU list
        order = ['FLUX', 'STDDEV', 'UNCORRECTED_FLUX', 'UNCORRECTED_STDDEV',
                 'LAMBDA', 'UNCORRECTED_LAMBDA', 'XS', 'YS', 'RA', 'DEC',
                 'ATRAN', 'RESPONSE', 'UNSMOOTHED_ATRAN']
        hdul = fits.HDUList()
        hdul.append(self.hdul[0])
        for hdu_name in order:
            hdul.append(self.hdul[hdu_name])
        self.hdul = hdul

        return self.hdul

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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            primary_header = self.create_primary_header(
                ra=ra, dec=dec,
                site_latitude=site_latitude, site_longitude=site_longitude,
                date_obs=date_obs, header_options=header_options)

        primary_hdu = fits.PrimaryHDU(header=primary_header)
        self.primary_header = primary_hdu.header
        self.hdul.append(primary_hdu)

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
        # Assume instrument is configured from header
        self.info.configuration.read_fits(self.primary_header)
        self.info.instrument.set_configuration(self.info.configuration)
        self.info.instrument.apply_configuration()
        self.set_source(ra=ra, dec=dec, header_options=header_options)
        self.update_header_scanning(self.primary_header)
        self.set_times(date_obs, header_options=header_options)
        self.set_start_site(site_longitude, site_latitude,
                            header_options=header_options)
        self.initialize_aircraft()
        self.update_header_weather(self.primary_header)
        self.update_header_origin(self.primary_header)
        self.create_source_model()
        self.info.configuration.read_fits(self.primary_header)
        self.info.detector_array.set_configuration(self.info.configuration)
        self.info.detector_array.apply_configuration()
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
            ra = get_hms_time(ra, angle=True)
            dec = get_dms_angle(dec)
            center = EquatorialCoordinates([ra, dec])
        else:
            center = EquatorialCoordinates([ra, dec])

        if 'OBSLAM' not in self.primary_header:
            self.update_header_value(
                self.primary_header, 'OBSLAM', center.ra.to('degree').value)
        if 'OBSBET' not in self.primary_header:
            self.update_header_value(
                self.primary_header, 'OBSBET', center.dec.to('degree').value)
        if 'SKY_ANGL' not in self.primary_header:
            self.update_header_value(self.primary_header, 'SKY_ANGL', 0.0)
        if 'DET_ANGL' not in self.primary_header:
            self.update_header_value(self.primary_header, 'DET_ANGL', 0.0)
        if 'DLAM_MAP' not in self.primary_header:
            self.update_header_value(self.primary_header, 'DLAM_MAP', 0.0)
        if 'DBET_MAP' not in self.primary_header:
            self.update_header_value(self.primary_header, 'DBET_MAP', 0.0)
        if 'OBSRA' not in self.primary_header:
            self.update_header_value(
                self.primary_header, 'OBSRA', center.ra.to('hourangle').value)
        if 'OBSDEC' not in self.primary_header:
            self.update_header_value(
                self.primary_header, 'OBSDEC', center.dec.to('degree').value)

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

        if ('SRCSIZE' not in self.primary_header or
                'SRCZSIZE' not in self.primary_header):
            self.info.configuration.read_fits(self.primary_header)
            self.info.instrument.apply_configuration()

        if 'SRCSIZE' not in self.primary_header:
            fwhm = self.info.instrument.xy_resolution.to('arcsec').value
            if source_type == 'extended':
                fwhm *= 3
            self.update_header_value(self.primary_header, 'SRCSIZE', fwhm)

        if 'SRCZSIZE' not in self.primary_header:
            z_fwhm = self.info.instrument.z_resolution.to('um').value
            self.update_header_value(self.primary_header, 'SRCZSIZE', z_fwhm)

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
        if header_options is not None and 'DATE-OBS' in header_options:
            timestamp = header_options['DATE-OBS']

        self.start_utc = DateRange.to_time(timestamp)
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

    def generate_scan_pattern(self):
        """
        Create the data HDU.

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

        detector = self.info.detector_array
        boresight_xy = detector.equatorial_to_detector_coordinates(equatorial)
        sx = boresight_xy.x[:, None] + detector.pixel_offsets.x[None]
        sy = boresight_xy.y[:, None] + detector.pixel_offsets.y[None]
        sxy = Coordinate2D([sx, sy])
        equatorial = detector.detector_coordinates_to_equatorial(sxy)

        spexel_line = np.ones(detector.n_spexel)[None, :, None]
        sx = sx[:, None] * spexel_line
        sy = sy[:, None] * spexel_line
        ra = equatorial.ra[:, None] * spexel_line
        dec = equatorial.dec[:, None] * spexel_line
        ra_hdu = fits.ImageHDU(data=ra.to('hourangle').value)
        ra_hdu.header['BUNIT'] = 'hourangle'
        ra_hdu.header['EXTNAME'] = 'RA'
        dec_hdu = fits.ImageHDU(data=dec.to('degree').value)
        dec_hdu.header['BUNIT'] = 'degree'
        dec_hdu.header['EXTNAME'] = 'DEC'
        xs_hdu = fits.ImageHDU(data=sx.to('arcsec').value)
        xs_hdu.header['BUNIT'] = 'arcsec'
        xs_hdu.header['EXTNAME'] = 'XS'
        ys_hdu = fits.ImageHDU(data=sy.to('arcsec').value)
        ys_hdu.header['BUNIT'] = 'arcsec'
        ys_hdu.header['EXTNAME'] = 'YS'
        self.hdul.append(xs_hdu)
        self.hdul.append(ys_hdu)
        self.hdul.append(ra_hdu)
        self.hdul.append(dec_hdu)

    def create_lambda_hdu(self):
        """
        Generate simulated wavelengths for each spexel/spaxel

        Returns
        -------
        wavelengths : numpy.ndarray
            The wavelengths of shape (n_spexel, n_spaxel) in um.
        """
        channel = self.info.instrument.options.get_string('CHANNEL')[0].upper()
        dichroic = self.info.instrument.options.get_int('DICHROIC')
        b_order = self.info.instrument.options.get_int('G_ORD_B')
        blue = channel == 'B'
        if blue:
            ind_pos = self.primary_header['G_STRT_B']
            gamma = 8.90080e-03
            if b_order == 1:
                g0 = 0.082656173
                n_p = 13.71363235
                a = 893.8674340
                ps = 0.000557523
                q_off = 6.753484718
                qs = 9.65511e-06
                is_off = [1075575.308, 1075337.272, 1075342.613, 1075360.888,
                          1075111.809, 1075254.097, 1075073.603, 1075093.902,
                          1075086.306, 1074898.3, 1075135.988, 1074975.497,
                          1074936.51, 1074950.201, 1074819.995, 1075073.411,
                          1074933.282, 1074883.507, 1074909.576, 1074778.233,
                          1075332.782, 1075128.462, 1075090.264, 1075092.068,
                          1074918.316]
            else:
                g0 = 0.082661321
                n_p = 14.62576144
                a = 829.1569652
                ps = 0.000560411
                q_off = 6.501570725
                qs = 7.24425e-06
                is_off = [1075354.214, 1075246.789, 1075259.151, 1075273.839,
                          1074986.764, 1075081.454, 1074995.319, 1075034.309,
                          1075025.152, 1074767.672, 1074989.908, 1074898.53,
                          1074874.335, 1074890.415, 1074680.646, 1074980.911,
                          1074884.451, 1074849.728, 1074851.092, 1074648.155,
                          1075238.121, 1075110.351, 1075046.954, 1075047.516,
                          1074799.607]
        else:
            gamma = 1.67200e-02
            ind_pos = self.primary_header['G_STRT_R']
            if dichroic == 105:
                g0 = 0.117162817
                n_p = 14.36394902
                a = 424.4860714
                ps = 0.000583733
                q_off = 6.229685338
                qs = 1.96254e-06
                is_off = [1150987.175, 1151124.26, 1151301.34, 1151461.228,
                          1151642.714, 1150351.927, 1150410.522, 1150506.261,
                          1150597.463, 1150766.426, 1149974.66, 1149970.751,
                          1149995.529, 1150002.171, 1150108.704, 1150237.575,
                          1150141.834, 1150073.606, 1150001.949, 1150101.825,
                          1150937.033, 1150796.478, 1150644.401, 1150487.3,
                          1150510.308]
            else:
                g0 = 0.117149497
                n_p = 14.30394914
                a = 426.3720058
                ps = 0.000587243
                q_off = 5.779980322
                qs = 1.31949e-06
                is_off = [1151352.872, 1151475.763, 1151659.806, 1151824.462,
                          1152007.864, 1150718.944, 1150776.749, 1150882.59,
                          1150966.167, 1151119.493, 1150343.198, 1150334.805,
                          1150364.927, 1150378.29, 1150464.099, 1150582.145,
                          1150489.52, 1150433.931, 1150360.028, 1150446.496,
                          1151271.944, 1151135.899, 1150986.77, 1150841.783,
                          1150839.159]

        # axis 0
        pix = np.arange(1, 17)
        sign = 2 * (((pix - q_off) > 0) - 0.5)
        delta = ((pix - 8.5) * ps)
        delta += sign * qs * (pix - q_off) ** 2

        # axis 1
        isf = 1.0
        is_off = np.asarray(is_off)
        modules = np.arange(25)
        slitpos = 25 - (6 * (modules // 5)) + (modules % 5)
        g = g0 * np.cos(np.arctan((slitpos - n_p) / a))
        phi = 2 * np.pi * isf * (ind_pos + is_off) / (2 ** 24)

        # cross terms for wavelength
        w = np.sin(np.add.outer(delta + gamma, phi))
        w += np.sin(phi - gamma)[None]
        w *= 1000 * g[None]

        # Manual correction to get wavecent in the middle
        w -= np.mean(w)
        w += self.primary_header['WAVECENT']

        hdu = fits.ImageHDU(data=w)
        hdu.header['EXTNAME'] = 'LAMBDA'
        hdu.header['BUNIT'] = 'um'
        self.hdul.append(hdu)

        dw_bary = self.primary_header.get('BARYSHIFT', 0.0)
        uw = w / (1 + dw_bary)
        hdu = fits.ImageHDU(data=uw)
        hdu.header['EXTNAME'] = 'UNCORRECTED_LAMBDA'
        hdu.header['BUNIT'] = 'um'
        self.hdul.append(hdu)

    def create_atran_response_hdu(self):
        """
        Create the ATRAN and TRANSMISSION HDUs.

        Returns
        -------
        None
        """
        min_wave = min(self.hdul['LAMBDA'].data.min(),
                       self.hdul['UNCORRECTED_LAMBDA'].data.min())
        max_wave = max(self.hdul['LAMBDA'].data.max(),
                       self.hdul['UNCORRECTED_LAMBDA'].data.max())
        shape = self.hdul['LAMBDA'].data.shape
        atran = np.full(shape, 0.95)
        response = np.full(shape, 4e-12)
        wave = np.linspace(min_wave, max_wave, 100)
        unsmoothed_atran = np.full_like(wave, atran.mean())
        hdu = fits.ImageHDU(data=atran)
        hdu.header['BUNIT'] = ''
        hdu.header['EXTNAME'] = 'ATRAN'
        self.hdul.append(hdu)
        hdu = fits.ImageHDU(data=response)
        hdu.header['BUNIT'] = 'adu/(Hz s Jy)'
        hdu.header['EXTNAME'] = 'RESPONSE'
        self.hdul.append(hdu)
        hdu = fits.ImageHDU(data=np.stack([wave, unsmoothed_atran]))
        hdu.header['BUNIT'] = ''
        hdu.header['EXTNAME'] = 'UNSMOOTHED_ATRAN'
        self.hdul.append(hdu)

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
        if self.info.instrument.channel.upper().startswith('B'):
            color = 'BLU'
        else:
            color = 'RED'

        date = header.get('DATE-OBS', self.start_utc.isot).split('T')[0]

        if 'FILENAME' not in header:
            filename = f'F0999_FI_IFS_09912345_{color}_WSH_00001.fits'
            self.update_header_value(header, 'FILENAME', filename)

        if 'OBS_ID' not in header:
            obs_id = f'P_{date}_FI_F999{color[0]}00001'
            self.update_header_value(header, 'OBS_ID', obs_id)

        if 'PLANID' not in header:
            header['PLANID'] = self.default_value('PLANID')

        if 'MISSN-ID' not in header:
            self.update_header_value(header, 'MISSN-ID',
                                     f'{date}_FI_F999')

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
            scan_length = 45.0

        dt = self.info.instrument.sampling_interval.to('second').value
        n_frames = scan_length / dt
        if n_frames % 1 != 0:
            n_frames = int(np.ceil(n_frames))
            scan_length = n_frames * dt

        for (key, default) in [
                ('EXPTIME', scan_length), ('TOTTIME', scan_length),
                ('OBSMODE', 'Scan'), ('SCNPATT', 'Daisy'),
                ('SCNCRSYS', 'TARF'), ('SCNITERS', 1), ('SCNANGLS', 0.0),
                ('SCNANGLC', 0.0), ('SCNANGLF', 0.0), ('SCNTWAIT', 0.0),
                ('SCNTRKON', 0), ('SCNRATE', 100.0), ('SCNCONST', True)]:
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
        elif x is not None and y is not None:
            pass
        else:
            width = source_size * 5
            if extended:
                width /= 2
            x = y = width

        constant_speed = header.get('SCNCONST', False)
        self.update_header_value(header, 'SCNCONST', constant_speed)
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

        constant_speed = header.get('SCNCONST', False)
        self.update_header_value(header, 'SCNCONST', constant_speed)
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
        constant_speed = self.primary_header['SCNCONST']
        return daisy_pattern_equatorial(self.source_equatorial,
                                        self.info.sampling_interval,
                                        n_oscillations=n_oscillations,
                                        radius=radius,
                                        radial_period=radial_period,
                                        constant_speed=constant_speed)

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

        constant_speed = self.primary_header['SCNCONST']
        equatorial = lissajous_pattern_equatorial(
            self.source_equatorial,
            self.info.sampling_interval,
            width=width, height=height, delta=delta, ratio=ratio,
            n_oscillations=n_oscillations,
            oscillation_period=oscillation_period,
            constant_speed=constant_speed)

        return equatorial

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
            model_name = 'single_gaussian_2d1'
            fwhm = self.primary_header.get('SRCSIZE') * units.Unit('arcsec')
            z_fwhm = self.primary_header.get('SRCZSIZE') * units.Unit('um')
            z_mean = self.primary_header.get('WAVECENT') * units.Unit('um')
            self.source_model = SimulatedSource.get_source_model(
                model_name, fwhm=fwhm, z_fwhm=z_fwhm, z_mean=z_mean)
        else:
            raise ValueError(
                f"{source_type} simulated source is not implemented.")

    def create_simulated_data(self):
        """
        Populate the FLUX values for the data HDU.

        Returns
        -------
        None
        """
        arcsec = units.Unit('arcsec')
        um = units.Unit('um')
        xs = self.hdul['XS'].data * arcsec
        ys = self.hdul['YS'].data * arcsec
        detector = self.info.detector_array
        xy_offsets = detector.detector_coordinates_to_equatorial_offsets(
            Coordinate2D([xs, ys]))

        z = np.empty(xs.shape, dtype=float) * um
        z[:] = self.hdul['LAMBDA'].data * um
        offsets = Coordinate2D1(xy=xy_offsets, z=z)

        source_data = self.source_model(offsets)
        if isinstance(source_data, units.Quantity):
            source_data = source_data.value

        source_amplitude = self.primary_header.get('SRCAMP', 100.0)
        source_data *= source_amplitude
        s2n = self.primary_header.get('SRCS2N', 20.0)
        rand = np.random.RandomState(0)  # Seed for testing
        noise = rand.randn(*source_data.shape)
        noise_level = source_amplitude / s2n
        noise *= noise_level
        source_data += noise

        hdu = fits.ImageHDU(data=source_data)
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['EXTNAME'] = 'UNCORRECTED_FLUX'
        self.hdul.append(hdu)

        stddev = np.full(source_data.shape, noise_level)
        hdu = fits.ImageHDU(data=stddev)
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['EXTNAME'] = 'UNCORRECTED_STDDEV'
        self.hdul.append(hdu)

        atran = self.hdul['ATRAN'].data
        source_data /= atran[None]
        hdu = fits.ImageHDU(data=source_data)
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['EXTNAME'] = 'FLUX'
        self.hdul.append(hdu)

        stddev /= atran[None]
        hdu = fits.ImageHDU(data=stddev)
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['EXTNAME'] = 'STDDEV'
        self.hdul.append(hdu)

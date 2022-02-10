# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re
import enum

from sofia_redux.scan.custom.sofia.flags.quality_flags import QualityFlags
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaProcessingInfo']


class SofiaProcessingInfo(InfoBase):

    flagspace = QualityFlags

    process_level_comment = {
        0: "Unknown processing level.",
        1: "Raw engineering/diagnostic data.",
        2: "Raw uncalibrated science data.",
        3: "Corrected/reduced science data.",
        4: "Flux-calibrated science data.",
        5: "Higher order product (e.g. composites)."
    }

    def __init__(self):
        super().__init__()
        self.associated_aors = None
        self.associated_mission_ids = None
        self.associated_frequencies = None
        self.process_level = None
        self.header_status = None
        self.software_name = None
        self.software_full_version = None
        self.product_type = None
        self.revision = None
        self.n_spectra = -1
        self.quality_level = self.flagspace.default_quality
        self.quality = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'proc'

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.process_level = options.get_string("PROCSTAT")
        self.header_status = options.get_string("HEADSTAT")
        self.quality = options.get_string("DATAQUAL")
        self.n_spectra = options.get_int("N_SPEC", default=-1)
        self.software_name = options.get_string("PIPELINE")
        self.software_full_version = options.get_string("PIPEVERS")
        self.product_type = options.get_string("PRODTYPE")
        self.revision = options.get_string("FILEREV")

        text = options.get_string("ASSC_AOR")
        if isinstance(text, str):
            self.associated_aors = [
                str(s).strip() for s in text.split(',') if len(s) > 0]
        else:
            self.associated_aors = None

        text = options.get_string("ASSC_MSN")
        if isinstance(text, str):
            self.associated_mission_ids = [
                str(s).strip() for s in re.split(r'[\t,]', text) if len(s) > 0]
        else:
            self.associated_mission_ids = None

        text = options.get_string("ASSC_FRQ")
        if isinstance(text, str):
            self.associated_frequencies = [
                float(s) for s in re.split(r'[\t, ]', text) if len(s) > 0]
        else:
            self.associated_frequencies = None

        level = str(self.quality).strip().upper()
        try:
            self.quality_level = self.flagspace.convert_flag(level)
        except AttributeError:
            self.quality_level = self.flagspace.default_quality

    @staticmethod
    def get_product_type(dims):
        if isinstance(dims, (int, float)):
            if dims == 0:
                return "HEADER"
            elif dims == 1:
                return "1D"
            elif dims == 2:
                return "IMAGE"
            elif dims == 3:
                return "CUBE"
            elif dims > 3:
                return f"{dims}D"
            else:
                return "UNKNOWN"
        else:
            return "UNKNOWN"

    @staticmethod
    def get_level_name(level):
        return f"LEVEL_{level}"

    def get_comment(self, level):
        """
        Return the FITS header comment for a given processing level.

        Parameters
        ----------
        level : str or int or enum
            The processing level.

        Returns
        -------
        comment : str
        """
        if level is None:
            level = 0
        elif isinstance(level, enum.Enum):
            level = level.value
        elif isinstance(level, str) and level.lower().startswith('level'):
            level = int(level[-1])
        else:
            level = int(level)
        if level < 0 or level > 4:
            return f"Invalid processing level: {level}"
        else:
            return self.process_level_comment[level]

    def get_processing(self, is_calibrated, dims, quality_level):
        """
        Return an updated processing information object.

        Parameters
        ----------
        is_calibrated : bool
            If the data are calibrated.
        dims : int
            The number of dimensions in the data.
        quality_level : int
            The quality level of the data.

        Returns
        -------
        SofiaProcessingInfo
        """
        result = SofiaProcessingInfo()
        result.process_level = self.get_level_name(3 if is_calibrated else 2)
        result.header_status = self.flagspace.convert_flag('MODIFIED').name
        result.software_name = "sofscan"
        # result.software_full_version = Awe.get_full_version()
        result.product_type = f"sofscan-{self.get_product_type(dims)}"
        result.quality_level = self.flagspace.convert_flag(quality_level)
        result.quality = result.quality_level.name.lower()
        return result

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        info = [('COMMENT', "<------ SOFIA Processing Data ------>")]
        if self.process_level is not None:
            info.append(('PROCSTAT', self.process_level,
                         self.get_comment(self.process_level)))

        if self.header_status is not None:
            info.append(('HEADSTAT', self.header_status, 'Header state.'))

        if self.software_name is not None:
            info.append(('PIPELINE', self.software_name,
                         'Software that created this file.'))

        if self.software_full_version is not None:
            info.append(('PIPEVERS', self.software_full_version,
                         'Full software version info.'))

        if self.product_type is not None:
            info.append(('PRODTYPE', self.product_type, 'Type of product.'))

        if self.revision is not None:
            info.append(('FILEREV', self.revision,
                         'File revision identifier.'))

        if self.quality is not None:
            info.append(('DATAQUAL', self.quality, 'Data quality level.'))

        if self.n_spectra > 0:
            info.append(('N_SPEC', self.n_spectra,
                         'Number of spectra included.'))

        if self.associated_aors is not None and len(self.associated_aors) > 0:
            info.append(('ASSC_AOR', ', '.join(self.associated_aors),
                         'Associated AOR IDs.'))

        if (self.associated_mission_ids is not None
                and len(self.associated_mission_ids) > 0):
            info.append(('ASSC_MSN', ', '.join(self.associated_mission_ids),
                         'Associated Mission IDs.'))

        if (self.associated_frequencies is not None
                and len(self.associated_frequencies) > 0):
            freqs = ', '.join([str(x) for x in self.associated_frequencies
                              if x is not None])
            info.append(('ASSC_FRQ', freqs, 'Associated Frequencies.'))

        insert_info_in_header(header, info, delete_special=True)

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'q':
            return self.quality_level
        elif name == 'nspec':
            return self.n_spectra
        elif name == 'quality':
            return self.quality
        elif name == 'level':
            return self.process_level
        elif name == 'stat':
            return self.header_status
        elif name == 'product':
            return self.product_type
        else:
            return super().get_table_entry(name)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.observation import ObservationInfo
from sofia_redux.scan.utilities.utils import insert_info_in_header

__all__ = ['SofiaObservationInfo']


class SofiaObservationInfo(ObservationInfo):

    def __init__(self):
        super().__init__()
        self.data_source = None
        self.obs_type = None
        self.source_type = None
        self.dictionary_version = None
        self.obs_id = None
        self.image_id = None
        self.aot_id = None
        self.aor_id = None
        self.file_group_id = None
        self.red_group_id = None
        self.blue_group_id = None
        self.is_primary_obs_id = False

    def apply_configuration(self):
        """
        Read and apply the configuration.

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return
        self.data_source = options.get_string("DATASRC")
        self.obs_type = options.get_string("OBSTYPE")
        self.source_type = options.get_string("SRCTYPE")
        self.dictionary_version = options.get_string("KWDICT")
        self.obs_id = options.get_string("OBS_ID")
        self.image_id = options.get_string("IMAGEID")
        self.aot_id = options.get_string("AOT_ID")
        self.aor_id = options.get_string("AOR_ID")
        self.project = self.aor_id
        self.file_group_id = options.get_string("FILEGPID")
        self.red_group_id = options.get_string("FILEGP_R")
        self.blue_group_id = options.get_string("FILEGP_B")
        self.set_source(options.get_string("OBJECT"))

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
        obs_id = f'P_{self.obs_id}' if self.is_primary_obs_id else self.obs_id
        info = [
            ('COMMENT', "<------ SOFIA Object Data ------>"),
            ('OBJECT', self.source_name, 'Object catalog name.'),
            ('DATASRC', self.data_source, 'Data source category.'),
            ('OBSTYPE', self.obs_type, 'Type of observation.'),
            ('SRCTYPE', self.source_type, 'AOR source type.'),
            ('KWDICT', self.dictionary_version,
             'SOFIA keyword dictionary version'),
            ('OBS_ID', obs_id, 'SOFIA observation ID.'),
        ]

        if self.image_id is not None:
            info.append(('IMAGEID', self.image_id,
                         'Image ID within an observation.'))

        if self.aot_id is not None:
            info.append(('AOT_ID', self.aot_id,
                         'Unique Astronomical Observation Template ID.'))

        if self.aor_id is not None:
            info.append(('AOR_ID', self.aor_id,
                         'Unique Astronomical Observation Request ID.'))

        if self.file_group_id is not None:
            info.append(('FILEGPID', self.file_group_id,
                         'User ID for grouping files together.'))

        if self.red_group_id is not None:
            info.append(('FILEGP_R', self.red_group_id,
                         'User ID for grouping red filter files together.'))

        if self.blue_group_id is not None:
            info.append(('FILEGP_B', self.blue_group_id,
                         'User ID for grouping blue filter files together.'))

        insert_info_in_header(header, info, delete_special=True)

    def is_aor_valid(self):
        """
        Checks whether the observation AOR ID is valid.

        Returns
        -------
        valid : bool
        """
        return self.valid_header_value(self.aor_id)

    def get_table_entry(self, name):
        """
        Given a name, return the parameter stored in the information object.

        Note that names do not exactly match to attribute names.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        if name == 'aor':
            return self.aor_id
        elif name == 'aot':
            return self.aor_id
        elif name == 'obsid':
            return self.obs_id
        elif name == 'src':
            return self.data_source
        elif name == 'dict':
            return self.dictionary_version
        elif name == 'fgid':
            return self.file_group_id
        elif name == 'imgid':
            return self.image_id
        elif name == 'obj':
            return self.source_name
        elif name == 'objtype':
            return self.source_type
        else:
            return super().get_table_entry(name)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
import numpy as np
import os
import pandas as pd

from sofia_redux.scan.channels.channels import Channels
from sofia_redux.scan.channels.mode.coupled_mode import CoupledMode
from sofia_redux.scan.channels.gain_provider.sky_gradient import SkyGradient
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.channels.rotating.rotating import Rotating

__all__ = ['Camera']


class Camera(Channels):

    def __init__(self, name=None, parent=None, info=None, size=0):
        """
        The Camera class contains additional rotation functionality.
        """
        super().__init__(name=name, parent=parent, info=info, size=size)

    @property
    def rotation(self):
        """
        Return the camera rotation angle.

        Returns
        -------
        angle : units.Quantity
        """
        return self.info.instrument.rotation

    @rotation.setter
    def rotation(self, value):
        """
        Set the camera rotation angle

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.info.instrument.rotation = value

    def init_modalities(self):
        """
        Initializes channel modalities.

        A modality is based of a channel division and contains a mode for each
        channel group in the channel division.

        The Camera modalities adds an additional CorrelatedModality to the
        existing modalities based on sky gradient in the x and y directions,
        coupled to the observing channels.

        Returns
        -------
        None
        """
        super().init_modalities()
        common_mode = self.modalities.get('obs-channels')[0]
        gain_provider_x = SkyGradient.x()
        gain_provider_y = SkyGradient.y()

        sky_x_mode = CoupledMode(common_mode, gain_provider=gain_provider_x)
        sky_x_mode.name = 'gradients:x'
        sky_y_mode = CoupledMode(common_mode, gain_provider=gain_provider_y)
        sky_y_mode.name = 'gradients:y'
        gradients = CorrelatedModality(name='gradients', identity='G')
        gradients.modes = [sky_x_mode, sky_y_mode]
        self.add_modality(gradients)

    def set_reference_position(self, reference):
        """
        Subtract a reference position from pixel positions.

        Parameters
        ----------
        reference : Coordinate2D
            The reference position containing (x, y) coordinates.

        Returns
        -------
        None
        """
        pixels = self.get_pixels()  # The channel data
        pixels.position.subtract(reference)

    def load_channel_data(self):
        """
        Load the channel data.

        The camera class implements rotation, which is performed before loading
        the standard channel data.  Rotation specified in the configuration is
        applied to the calculated or default positions only.  RCP rotation is
        handled separately via the 'rcp.rotate' configuration option.
        Instruments with a rotator should apply explicit rotation after pixel
        positions are finalized.

        Returns
        -------
        None
        """
        self.rotation = 0.0 * units.Unit('deg')
        if 'rotation' in self.configuration:
            self.rotate(
                self.configuration.get_float('rotation') * units.Unit('deg'))

        rcp_file = self.configuration.get_string('rcp', default=None)
        if rcp_file is not None:
            rcp_file = self.configuration.get_filepath('rcp', default=None)
            if rcp_file is not None:
                self.read_rcp(rcp_file)
            else:
                log.warning("Cannot update pixel RCP data. Using values from "
                            "FITS.")

        if isinstance(self, Rotating):
            angle = self.get_rotation()
            if angle != 0.0:
                self.rotate(angle)

        super().load_channel_data()

    def get_rotation_angle(self):
        """
        Return the rotation angle.

        Returns
        -------
        angle : astropy.units.Quantity
        """
        return self.info.get_rotation_angle()

    @staticmethod
    def get_rcp_info(filename):
        """
        Return the RCP file information as a `pandas` data frame.

        If the RCP file contains 3 columns, x and y positions are returned.
        If there are 5 or more columns, the columns are considered to be
        [source gain, coupling, x, y].  The zeroth column gives the channel
        index.

        Parameters
        ----------
        filename : str
            Path to the RCP file.

        Returns
        -------
        rcp_information : pandas.DataFrame or None
            The RCP information.  `None` will be returned on failure.
        """
        # This is a test file:
        # /Users/dperera/mmoc_repo/awe/mine/resources/original/aszca/master.rcp

        if not isinstance(filename, str) or not os.path.isfile(filename):
            return None

        log.info(f"Reading RCP from {filename}")

        df = pd.read_csv(filename, comment='#', delim_whitespace=True,
                         header=None, converters={0: lambda x: str(x)})
        n_columns = df.columns.size
        df = df.set_index([0])

        if n_columns == 3:
            column_names = {1: 'x', 2: 'y'}
        elif n_columns == 4:
            column_names = {1: 'gain', 2: 'x', 3: 'y'}
        elif n_columns > 4:
            column_names = {1: 'gain', 2: 'coupling', 3: 'x', 4: 'y'}
        else:
            log.error("Invalid number of columns in RCP file.")
            return None

        df = df.rename(columns=column_names)
        return df

    def read_rcp(self, filename):
        """
        Read and apply the RCP file information to channels (pixels).

        The RCP information is read and applied from a given file.  The RCP
        file should contain comma-separated values in one of following column
        formats:

        CHANNEL_INDEX, X_POSITION(arcsec), Y_POSITION(arcsec)
        CHANNEL_INDEX, GAIN, X_POSITION(arcsec), Y_POSITION(arcsec)
        CHANNEL_INDEX, GAIN, COUPLING, X_POSITION(arcsec), Y_POSITION(arcsec)

        All pixels not contained in the RCP file are flagged as BLIND, and will
        only be unflagged if a GAIN column is available in the file.  The
        channel coupling will be set to GAIN/COUPLING or GAIN/channel.gain
        depending on the column format, or ignored if not available.  X and Y
        positions are also set at this stage.

        If no RCP information is available (no file), these attributes should
        be set via other methods.

        Parameters
        ----------
        filename : str
            Path to the RCP file.

        Returns
        -------
        None
        """
        rcp_df = self.get_rcp_info(filename)
        if rcp_df is None:
            log.warning("Cannot update pixel RCP data. Using values from "
                        "FITS.")

        # Channels not in the RCP file are assumed to be blind
        self.data.set_flags('BLIND')
        use_gains = self.configuration.get_bool('rcp.gains')
        if use_gains:
            log.info("Initial source gains set from RCP file")

        arcsec = units.Unit('arcsec')

        source_gain = rcp_df['gain'].values if 'gain' in rcp_df else None
        coupling = rcp_df['coupling'].values if 'coupling' in rcp_df else None
        x = rcp_df['x'].values * arcsec if 'x' in rcp_df else None
        y = rcp_df['y'].values * arcsec if 'y' in rcp_df else None
        fixed_indices = rcp_df.index.astype(int).values
        indices = self.data.find_fixed_indices(fixed_indices)

        if source_gain is not None:
            if use_gains:
                if coupling is None:
                    base_gain = self.data.gain[indices]
                    nzi = base_gain != 0
                    coupling = source_gain.copy()
                    coupling[nzi] /= base_gain[nzi]
                    coupling[~nzi] = 0.0
                    self.data.coupling[indices] = coupling
                else:
                    self.data.coupling[indices] = source_gain / coupling

            unflag_mask = source_gain != 0
            self.data.unflag(indices=indices[unflag_mask],
                             flag='BLIND')

        position = self.data.position[indices]
        if x is not None:
            position.x = x
        if y is not None:
            position.y = y
        self.data.position[indices] = position

        self.flag_invalid_positions()

        rcp_center = self.configuration.float_list('rcp.center', default=None)
        if rcp_center is not None:
            self.data.position.x -= rcp_center[0] * arcsec
            self.data.position.y -= rcp_center[1] * arcsec

        rcp_rotate = self.configuration.get_float('rcp.rotate', default=None)
        if rcp_rotate is not None:
            rcp_rotate = rcp_rotate * units.Unit('deg')
            Coordinate2D.rotate_offsets(self.data.position, rcp_rotate)

        rcp_zoom = self.configuration.get_float('rcp.zoom', default=None)
        if rcp_zoom is not None:
            self.data.position.scale(rcp_zoom)

    def get_rcp_header(self):
        """
        Return the header for an RCP file.

        Returns
        -------
        header : str
        """
        return "ch\t[Gpnt]\t[Gsky]ch\t[dX\"]\t[dY\"]"

    def print_pixel_rcp(self, header=None):
        """
        Return string information of the pixel RCP info.

        Parameters
        ----------
        header : str, optional
            An optional header string.

        Returns
        -------
        pixel_rcp : str
        """
        msg = ['# SOFSCAN Receiver Channel Parameter (RCP) Data File.', '#']
        if header is not None:
            msg.append(header)
        msg.append('#')
        mapping_pixels = self.get_mapping_pixels(
            discard_flag=self.flagspace.sourceless_flags())

        positions = mapping_pixels.position
        keep_indices = np.nonzero(~positions.is_nan())[0]
        indices = mapping_pixels.indices[keep_indices]
        rcp_info = self.data.get_rcp_string(indices=indices).split('\n')
        msg.append(f'# {rcp_info[0]}')
        if len(rcp_info) > 1:
            msg.extend(rcp_info[1:])
        return '\n'.join(msg)

    def flag_invalid_positions(self, maximum_distance=1.0 * units.Unit('deg')):
        """
        Flag positions as blind if the distance is greater than a given limit.

        Parameters
        ----------
        maximum_distance : astropy.units.Quantity
            A spatial maximum distance limit.  The default is 1 degree.

        Returns
        -------
        None
        """
        pixels = self.get_pixels()
        length = pixels.position.length
        invalid = (length > maximum_distance) | pixels.position.is_nan()
        self.data.set_flags('BLIND', indices=invalid)

    def rotate(self, angle):
        """
        Apply a rotation to channel (pixel) positions.

        Parameters
        ----------
        angle : units.Quantity
            The rotation to apply.

        Returns
        -------
        None
        """
        if angle is None or np.isnan(angle):
            return

        log.debug(f"Applying {angle} rotation to channel positions.")

        # Undo prior rotation
        prior_offset = self.info.get_pointing_offset(self.rotation)
        new_offset = self.info.get_pointing_offset(self.rotation + angle)

        position = self.get_pixels().position

        # Center positions on the rotation center
        position.subtract(prior_offset)
        # Do the rotation
        position.rotate(angle)
        # Re-center on the pointing center
        position.add(new_offset)
        self.rotation += angle

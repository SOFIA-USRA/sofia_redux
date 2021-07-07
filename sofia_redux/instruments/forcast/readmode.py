# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits

from sofia_redux.instruments.forcast.chopnod_properties \
    import chopnod_properties
from sofia_redux.instruments.forcast.getpar import getpar

__all__ = ['readmode']


def readmode(header):
    """
    Read the chop/nod and instrument mode from the header

    Reads various parameters from the input header to determine the
    chop/nod and instrument mode of the observation.  Returns a
    single string indicating the mode used.  This function is
    required because the instrument mode definition became
    complicated with all these combinations of header keywords,
    depending on the origin and date of the data.  In order to be
    sure that the instrument mode is read the same way along the
    code, the mode should always be defined using this function.

    Procedure:

    1. We need to read instmode first because if the observation was C2
       the skymode is not updated to C2 but may be NMC or NPC. Thus, we
       don't consider skymode unless it is a C2N observation.

    2. Read skymode. All data after Basic Science should contain this
       keyword with the right C2NC2 value if this mode is used.

    3. If skymode is not found, then this means that the data is from
       Basic Science (or there is an error in the header) In that case,
       the INSTMODE keyword contains the instrument mode except for C2NC2
       that is defined by the presence of the C2NC2 keyword.

    4. It could happen for Basic Science data that there is no skymode,
       so the instmode is automatically set.  Thus, for C2N the value is
       not updated to NMC or NPC. Below, we check the nod and chop
       amplitudes and update mode accordingly.

    5. After basic science there was a change in the C2NC2 data. Instead
       of storing them in 2 plane data cubes that are either on or off
       source (A or B), the data is stored in 4 plane data cubes. These
       files contain both on and off positions.  The only way for now to
       know what sort of data is (for now) is to see the number of planes
       which is given by the NAXIS3 keyword.  We defined a virtual mode
       called C2NC4 to indicate a 4 plane C2NC2.  SKYMODE is updated
       in the header accordingly.

    6. Read the size of the 3rd axis. If the dimension is 2 then the it
       is an old C2NC2 which stored on-off-positions in two different
       files (C2 like).  If the dimension is 4 then the on-off-positions
       are in the same file as 4 planes

    Parameters
    ----------
    header : astropy.io.fits.header.Header

    Returns
    -------
    str or None
        The instrument mode (None if not found)
    """
    if not isinstance(header, fits.header.Header):
        log.error("invalid header (%s)" % type(header))
        return

    instmode = getpar(
        header, 'INSTMODE', dtype=str, default='NONE',
        update_header=False, dripconf=False).strip().upper()
    skymode = getpar(
        header, 'SKYMODE', dtype=str, default='NONE',
        update_header=False, dripconf=False).strip().upper()

    if instmode == 'C2' and int(header.get('C2NC2', 0)) == 1:
        mode = 'C2NC2'
    elif instmode == 'C2N':
        if skymode != 'NONE':
            mode = skymode
        else:
            mode = instmode
    elif skymode in ['C2NC2', 'C2NC4']:
        mode = skymode
    else:
        mode = instmode

    if mode == 'C2N':
        if chopnod_properties(header)['nmc']:
            mode = 'NMC'
        else:
            mode = 'NPC'
    elif mode == 'C2NC2':
        if int(header.get('NAXIS3', 0)) == 4:
            mode = 'C2NC4'
            if 'SKYMODE' in header:
                header['SKYMODE'] = 'C2NC4', header.comments['SKYMODE']
            else:
                header['SKYMODE'] = 'C2NC4'

    return mode

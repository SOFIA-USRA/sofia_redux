# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from collections.abc import Sized

from astropy import log
from astropy.io.fits.header import Header
import pandas

from sofia_redux.toolkit.utilities.fits import hdinsert, kref, href
from sofia_redux.toolkit.utilities.func import date2seconds

import sofia_redux.instruments.forcast as drip
from sofia_redux.instruments.forcast.getpar import getpar


from statistics import mean, median

__all__ = ['addparent', 'hdadd', 'hdkeymerge', 'hdmerge']


def addparent(name, header,
              comment="id or name of file used in the processing"):
    """
    Add an id or file name to a header as PARENTn

    Adds the ID or filename of an input file to a specified header array,
    under the keyword PARENTn, where n is some integer greater than 0.
    If a previous PARENTn keyword exists, n will be incremented to
    produce the new keyword.

    If no PARENTn keyword exists, a new card will be appended to the end
    of the header.  Otherwise, the card will be inserted after PARENT(n-1).

    Parameters
    ----------
    name : str
        Name or id of the file to be recorded in the header
    header : astropy.io.fits.header.Header
        Input header to be updated
    comment : str
        Comment for PARENTn

    Returns
    -------
    None
    """
    parents = header.cards['PARENT*']
    value = os.path.basename(name)
    if len(parents) == 0:
        hdinsert(header, 'PARENT1', value, comment=comment, refkey=kref)

    existing_values = [x[1] for x in parents]
    if value in existing_values:
        return

    existing_keys = [x[0] for x in parents]
    parentn = 1
    last_parent = kref
    while True:
        key = 'PARENT' + str(parentn)
        if key in existing_keys:
            last_parent = key
            parentn += 1
        else:
            break
    hdinsert(header, key, value,
             refkey=last_parent, comment=comment, after=True)


def hdadd(header, reference, refline, do_history=False):
    """
    Add keywords and values from one header to a reference header

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    reference : astropy.io.fits.header.Header
    refline : str
    do_history : bool

    Returns
    -------
    None
    """

    # initialize space for new information from the current header
    filename = header.get('FILENAME')
    if filename is not None:
        hdinsert(reference, 'HISTORY', '', refkey=href)
        hdinsert(reference, 'HISTORY',
                 '............................................',
                 refkey=href)
        hdinsert(reference, 'HISTORY', 'FILE: %s' % filename, refkey=href)
        hdinsert(reference, 'HISTORY', '', refkey=href)

    # Find the section of the current header corresponding to the
    # pipelined data.
    if refline not in header.values():
        return
    pipeline_start = [*header.values()].index(refline) + 2

    skip_alias = [
        'COMMEN', 'CONTIN', 'MRGDX0', 'MRGDY0', 'MRGDX1', 'MRGDY1',
        'MRGDX2', 'MRGDY2', 'SCRPIX', 'SCRVAL', 'SCROTA', 'SCDELT',
        'UCRPIX', 'UCRVAL', 'UCROTA', 'UCDELT', 'MCRPIX', 'MCRVAL',
        'MCROTA', 'MCDELT'
    ]

    for key, value, comment in header.cards[pipeline_start:]:
        keyalias = key.strip()[:6]
        keyname = key.strip()[:8]
        if keyalias in skip_alias:
            continue
        ncards = header.count(keyname)
        if keyalias not in ['COMMEN', 'HISTOR', 'CONTIN']:
            if ncards > 1:
                value = header[(keyname, 0)]
                comment = header.comments[(keyname, 0)]
            else:
                value = header[keyname]
                comment = header.comments[keyname]

        if keyalias == 'HISTOR':
            if do_history:
                hdinsert(reference, 'HISTORY', value, refkey=href)
        elif keyalias == 'PARENT':
            addparent(value, reference, comment=comment)
        elif keyalias == 'COADX0':
            val, coadkey = 1, 'COADX1'
            while coadkey in reference:
                coadkey = 'COADX%s' % (val + 1)
                val += 1
            hdinsert(reference, coadkey, value,
                     comment='X shift during coadd process')
        elif keyalias == 'COADY0':
            val, coadkey = 1, 'COADY1'
            while coadkey in reference:
                coadkey = 'COADY%s' % (val + 1)
                val += 1
            hdinsert(reference, coadkey, value,
                     comment='Y shift during coadd process')
        else:
            if keyname not in reference:
                hdinsert(reference, keyname, value, comment=comment)


def hdkeymerge(headers, reference, keyname, keytype):
    """Merge header keys according to rules.

    Called by hdmerge to merge particular keys according
    to rules outlined in a keyword definition table.

    Parameters
    ----------
    headers : Sequence of astropy.io.fits.header.Header
       List of headers to merge.
    reference : astropy.io.fits.header.Header
       Reference header.
    keyname : str
       Keyword name to merge.
    keytype : {'first', 'last', 'same', 'concatenate',
       'mean', 'median', 'sum', 'min', 'max'}
       Merge type.

    Returns
    -------
    None
    """
    ktype = keytype.lower().strip()
    if len(headers) > 1:
        if keytype == 'multidefflt':
            hdinsert(reference, keyname, -9999)
            return
        elif keytype == 'multidefstr':
            hdinsert(reference, keyname, 'UNKNOWN')
            return

    if ktype in ['first', 'last']:
        times = {}
        comment = None
        for header in headers:
            value = header.get(keyname)
            fname = header.get('FILENAME')
            if value is None:
                msg = "%s keyword is not present in %s" % (keyname, fname)
                log.debug(msg)
                hdinsert(reference, 'HISTORY', msg, refkey=href)
                continue
            seconds = None
            comment = header.comments[keyname]
            date = header.get('DATE')
            if date is None:
                msg = "DATE keyword is not present in %s" % fname
                log.warning(msg)
                hdinsert(reference, 'HISTORY', msg, refkey=href)
            else:
                seconds = date2seconds(date)
                if seconds is None:
                    msg = "DATE keyword is wrong in %s" % fname
                    log.warning(msg)
                    hdinsert(reference, 'HISTORY', msg, refkey=href)
            if seconds in times:
                times[seconds].append(value)
            else:
                times[seconds] = [value]
        timekeys = [t for t in times.keys() if t is not None]
        if len(timekeys) == 0:
            timekey = None
        else:
            timekey = min(timekeys) if ktype == 'first' else max(timekeys)
        if timekey in times:
            value = times[timekey][0]
            hdinsert(reference, keyname, value, comment=comment)
        else:
            msg = "%s keyword not present in header list" % keyname
            log.debug(msg)
            hdinsert(reference, 'HISTORY', msg, refkey=href)
        return

    values = [header.get(keyname) for header in headers]
    idxvalues = [(x[0], x[1]) for x in enumerate(values) if x[1] is not None]
    if len(idxvalues) == 0:
        msg = "%s keyword not present in header list" % keyname
        log.debug(msg)
        hdinsert(reference, 'HISTORY', msg, refkey=href)
        return

    values = [x[1] for x in idxvalues]
    comment = headers[idxvalues[-1][0]].comments[keyname]
    if ktype == 'same':
        val0 = headers[idxvalues[0][0]].get(keyname)
        fname0 = headers[idxvalues[0][0]].get("FILENAME")
        hdinsert(reference, keyname, val0, comment=comment)
        for idx, value in idxvalues[1:]:
            if value != val0:
                msg = "%s in %s does not match %s" % (
                    keyname, headers[idx].get('FILENAME'), fname0)
                log.debug(msg)

    elif ktype == 'concatenate':
        value = ','.join([str(val) for val in sorted(set(values))])
        hdinsert(reference, keyname, value, comment=comment)

    elif ktype in ['mean', 'median', 'sum', 'min', 'max']:
        valid_numbers = []
        for idx, value in enumerate(values):
            try:
                float(value)
            except ValueError:
                fname = headers[idxvalues[idx][0]].get("FILENAME")
                msg = "%s keyword in %s cannot be included in %s" % (
                    (keyname, fname, ktype))
                log.warning(msg)
                log.warning("Type is: %s" % type(value))
                hdinsert(reference, 'HISTORY', msg, refkey=href)
            else:
                valid_numbers.append(value)
        if len(valid_numbers) == 0:
            msg = "cannot take %s of %s keyword in " \
                  "header list" % (ktype, keyname)
            log.warning(msg)
            hdinsert(reference, 'HISTORY', msg, refkey=href)
        elif ktype == 'mean':
            hdinsert(reference, keyname, mean(valid_numbers),
                     comment=comment)
        elif ktype == 'median':
            val = median(valid_numbers)
            val = type(valid_numbers[0])(val)
            hdinsert(reference, keyname, val, comment=comment)
        elif ktype == 'min':
            hdinsert(reference, keyname, min(valid_numbers), comment=comment)
        elif ktype == 'max':
            hdinsert(reference, keyname, max(valid_numbers), comment=comment)
        elif ktype == 'sum':
            hdinsert(reference, keyname, sum(valid_numbers), comment=comment)
    else:
        return


def hdmerge(headers, reference_header,
            refline='--------- Pipeline related Keywords --------',
            fkeydef='output-key-merge.txt',
            hdinit=None):
    """
    Merge values from multiple headers

    This function looks for a reference line in a COMMENT keyword.
    Any keywords above this line are extracted from the inital header
    (HDINIT) and the reference header and combined.  Keywords after
    this line are extracted from all input headers and combined
    according to specified rules.  Generally, the rule is that
    keywords are taken from the inital header.  Any keywords that
    require special treatment should be defined in a separate file:
    output-key-merge.txt, stored in the calibration data directory.
    This file has two columns: name and type.  The possible types
    are:

        - same: the value should be the same in all headers
        - mean: the output value should be the mean of the input values
        - median : the output value should be the median of the input
                   values
        - min: the output value should be the minimum of the input values
        - max: the output value should be the maximum of the input values
        - first: the output value should be the value in the file with
                 the earliest date/time
        - last: the output value should be the value in the file with the
                latest date/time
        - multidefflt: the output should be the floating point default
                       value (-9999) if there are multiple input files
        - multdefstr: the output value should be the string default
                      value ('UNKNOWN') if there are multiple input files
        - concatenate: all unique values should be concatenated with a
                       comma
        - sum: all values should be added together

    Parameters
    ----------
    headers : Sequence of astropy.io.fits.header.Header
        List of input headers to merge
    reference_header : astropy.io.fits.header.Header
        Reference header
    refline : str
        If provided, this will be the value of the COMMENT keyword
        after which keywords should be treated as originating from the
        pipeline.  If not provided, the default value will be used.
    fkeydef : str
        File path to the definition file of how to handle output
        keywords.  If not specified, the default value will be used.
    hdinit : astropy.io.fits.header.Header
        Header to take values above the `refline` from.  If not
        specified, the refhead will be used.

    Returns
    -------
    astropy.io.fits.header.Header
        Merged header array
    """
    # Set the final output equal to the reference header combined with the
    # first elements of the headerlist
    if not isinstance(headers, Sized):
        log.error("invalid header list")
        return
    elif not all([isinstance(h, Header) for h in headers]):
        log.error("invalid header list")
        return
    elif not isinstance(reference_header, Header):
        log.error("invalid reference header")
        return
    elif hdinit is not None:
        if not isinstance(hdinit, Header):
            log.error("invalid hdinit header")
            return

    if isinstance(hdinit, Header):
        if refline in hdinit.values():
            idxlim = [*hdinit.values()].index(refline) - 1
        else:
            if len(headers) == 0:
                log.error("hdinit supplied with bad input arguments")
                return
            else:
                idxlim = len(headers[0])
        if refline in reference_header.values():
            k = [*reference_header.values()].index(refline)
            merged_header = hdinit.copy()[:(idxlim - 1)]
            merged_header += reference_header.copy()[(k - 1):]
        else:
            merged_header = hdinit.copy()
    else:
        merged_header = reference_header.copy()

    if len(headers) == 0:
        return merged_header

    hdadd(headers[0], merged_header, refline=refline, do_history=True)
    # If there is only one header in the headerlist there is no point
    # in going further
    if len(headers) == 1:
        return merged_header

    dohist = len(headers) <= 20
    for header in headers[1:]:
        hdadd(header, merged_header, refline=refline, do_history=dohist)

    if os.path.sep not in fkeydef:
        path = getpar(merged_header, 'CALDATA', default='UNKNOWN', dtype=str,
                      comment='path containing the ancillary data')
        if path == 'UNKNOWN':
            path = os.path.join(os.path.dirname(drip.__file__), 'data')

        if not os.path.isdir(path):
            log.error("hdrequirements: input file and configuration do "
                      "not define CALDATA directory")
            return merged_header
        fkeydef = os.path.join(path, 'output-key-merge.txt')

    if not os.path.isfile(fkeydef):
        log.error("Common key definition file does not exist: %s" % fkeydef)
        return merged_header

    keydef = pandas.read_csv(
        fkeydef, comment=';', delim_whitespace=True,
        names=['keyword', 'merge_type'])
    if len(keydef) == 0:
        log.error('No valid lines in common key definition file %s'
                  % fkeydef)
        return merged_header

    # Now we merge the keywords if specified by mergekeys
    for _, row in keydef.iterrows():
        hdkeymerge(headers, merged_header, row['keyword'], row['merge_type'])

    return merged_header

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

from sofia_redux.toolkit.utilities.fits import set_log_level


def test_log_level(capsys):
    # overall log level needs to be low enough to show messages

    # start with the log at a level that wouldn't
    # show any messages
    orig_level = log.level
    log.setLevel('CRITICAL')

    dbg = 'test debug'
    inf = 'test info'
    wrn = 'test warning'
    err = 'test error'
    with set_log_level('DEBUG'):
        log.debug(dbg)
        log.info(inf)
        log.warning(wrn)
        log.error(err)
        capt = capsys.readouterr()
        assert dbg in capt.out
        assert inf in capt.out
        assert wrn in capt.err
        assert err in capt.err
    with set_log_level('INFO'):
        log.debug(dbg)
        log.info(inf)
        log.warning(wrn)
        log.error(err)
        capt = capsys.readouterr()
        assert dbg not in capt.out
        assert inf in capt.out
        assert wrn in capt.err
        assert err in capt.err
    with set_log_level('WARNING'):
        log.debug(dbg)
        log.info(inf)
        log.warning(wrn)
        log.error(err)
        capt = capsys.readouterr()
        assert dbg not in capt.out
        assert inf not in capt.out
        assert wrn in capt.err
        assert err in capt.err
    with set_log_level('ERROR'):
        log.debug(dbg)
        log.info(inf)
        log.warning(wrn)
        log.error(err)
        capt = capsys.readouterr()
        assert dbg not in capt.out
        assert inf not in capt.out
        assert wrn not in capt.err
        assert err in capt.err
    with set_log_level('CRITICAL'):
        log.debug(dbg)
        log.info(inf)
        log.warning(wrn)
        log.error(err)
        capt = capsys.readouterr()
        assert dbg not in capt.out
        assert inf not in capt.out
        assert wrn not in capt.err
        assert err not in capt.err

    # reset the original log level
    log.setLevel(orig_level)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import enum
import numpy as np
import pytest

from sofia_redux.scan.flags.flags import Flags


class BasicFlags(Flags):
    class BasicFlagTypes(enum.Flag):
        FLAG1 = enum.auto()
        FLAG2 = enum.auto()
        FLAG3 = enum.auto()

    flags = BasicFlagTypes
    descriptions = {
        flags.FLAG1: 'flag 1',
        flags.FLAG2: 'flag 2',
        flags.FLAG3: 'flag 3'
    }
    letters = {
        '1': flags.FLAG1,
        '2': flags.FLAG2,  # deliberately missing flag 3
    }


flag1 = BasicFlags.flags.FLAG1
flag2 = BasicFlags.flags.FLAG2
flag3 = BasicFlags.flags.FLAG3


@pytest.fixture
def flags():
    return BasicFlags


def test_all_flags(flags):
    all_flags = flags.all_flags()
    assert flag1 & all_flags
    assert flag2 & all_flags
    assert flag3 & all_flags


def test_letter_to_flag(flags):
    assert flags.letter_to_flag('1') == flag1
    assert flags.letter_to_flag('2') == flag2
    assert flags.letter_to_flag('3').value == 0


def test_flag_to_letter(flags):
    assert flags.flag_to_letter(flag1) == '1'
    assert flags.flag_to_letter(flag2) == '2'
    assert flags.flag_to_letter(flag3) == '-'
    assert flags.flag_to_letter(flag1 | flag2) == '12'


def test_flag_to_description(flags):
    assert flags.flag_to_description(0) == ''
    assert flags.flag_to_description(1) == 'flag 1'
    assert flags.flag_to_description(7) == 'flag 1 & flag 2 & flag 3'


def test_parse_string(flags):
    assert flags.parse_string('1') == flag1
    assert flags.parse_string('2') == flag2
    assert flags.parse_string('12') == flag1 | flag2


def test_convert_flag(flags):
    assert flags.convert_flag(None).value == 0
    assert flags.convert_flag(flag1) == flag1
    assert flags.convert_flag(2) == flag2
    assert flags.convert_flag(-1) == flags.all_flags()
    assert flags.convert_flag('1') == flag1
    assert flags.convert_flag('1|2') == flag1 | flag2
    assert flags.convert_flag('flag1') == flag1
    assert flags.convert_flag('1|2|flag3') == flags.all_flags()
    with pytest.raises(ValueError) as err:
        flags.convert_flag(1.0)
    assert 'Invalid flag type' in str(err.value)


def test_is_flagged(flags):
    assert flags.is_flagged(flag1, flag=flag1)
    assert flags.is_flagged(1)
    assert not flags.is_flagged(1, flag=0)
    assert flags.is_flagged(0, flag=0)
    assert flags.is_flagged(1, flag=flags.all_flags())
    assert not flags.is_flagged(1, flag=flags.all_flags(), exact=True)
    assert flags.is_flagged(flags.all_flags(), flag=flags.all_flags(),
                            exact=True)
    assert flags.is_flagged(np.empty(0, dtype=int)).size == 0

    a = (np.random.random((10, 11)) >= 0.5).astype(int)
    mask = flags.is_flagged(a, indices=False)
    assert np.allclose(a[mask], 1)
    assert np.allclose(a[~mask], 0)
    assert mask.dtype == bool
    assert mask.shape == (10, 11)

    indices = flags.is_flagged(a, indices=True)
    assert np.allclose(a[indices], 1)
    assert isinstance(indices, tuple)
    assert indices[0].size == a.sum()


def test_is_unflagged(flags):
    assert not flags.is_unflagged(flag1, flag=flag1)
    assert not flags.is_unflagged(1)
    assert flags.is_unflagged(1, flag=0)
    assert not flags.is_unflagged(0, flag=0)
    assert not flags.is_unflagged(1, flag=flags.all_flags())
    assert flags.is_unflagged(1, flag=flags.all_flags(), exact=True)

    assert not flags.is_unflagged(flags.all_flags(), flag=flags.all_flags(),
                                  exact=True)
    assert flags.is_unflagged(np.empty(0, dtype=int)).size == 0
    a = (np.random.random((10, 11)) >= 0.5).astype(int)
    mask = flags.is_unflagged(a, indices=False)
    assert np.allclose(a[mask], 0)
    assert np.allclose(a[~mask], 1)
    assert mask.dtype == bool
    assert mask.shape == (10, 11)

    indices = flags.is_unflagged(a, indices=True)
    assert np.allclose(a[indices], 0)
    assert isinstance(indices, tuple)
    assert indices[0].size == a.size - a.sum()


def test_and_operation(flags):
    assert flags.and_operation(1, flag1)
    assert not flags.and_operation(flag1, flag2)
    a = np.full(5, 5)
    assert np.allclose(flags.and_operation(a, flag1), 1)
    assert np.allclose(flags.and_operation(a, flag2), 0)
    assert np.allclose(flags.and_operation(a, flag3), 4)


def test_or_operation(flags):
    assert flags.or_operation(1, flag3) == 5
    assert flags.or_operation(flag1, flag2) == 3
    a = np.full(5, 1)
    assert np.allclose(flags.or_operation(a, flags.all_flags()), 7)


def test_discard_mask(flags):
    test = np.arange(10)  # This should cover all flags and combos
    mask = flags.discard_mask(test, flag=None, criterion=None)
    assert not mask[0] and mask[1:].all()  # DISCARD_ANY
    mask = flags.discard_mask(test, flag=flag1, criterion='DISCARD_ANY')
    assert not mask[0] and mask[1:].all()  # flag is irrelevant
    mask = flags.discard_mask(test, flag=flag1, criterion='DISCARD_ALL')
    assert mask[1::2].all() and not mask[0::2].any()
    mask = flags.discard_mask(test, flag=flag1, criterion='DISCARD_MATCH')
    assert not mask[0] and mask[1] and not mask[2:].any()
    mask = flags.discard_mask(test, flag=flag1, criterion='KEEP_ANY')
    assert mask[0] and not mask[1:].any()  # flag is irrelevant
    mask = flags.discard_mask(test, flag=flag1, criterion='KEEP_ALL')
    assert mask[0::2].all() and not mask[1::2].any()
    mask = flags.discard_mask(test, flag=flag1, criterion='KEEP_MATCH')
    assert mask[0] and not mask[1] and mask[2:].all()
    with pytest.raises(ValueError) as err:
        _ = flags.discard_mask(test, flag=flag1, criterion='FOO')
    assert 'Invalid criterion flag' in str(err.value)


def test_flag_mask(flags):
    test = np.arange(10)  # This should cover all flags and combos
    mask = flags.flag_mask(test, flag=None, criterion=None)
    assert not mask[0] and mask[1:].all()  # KEEP_ANY
    mask = flags.flag_mask(test, flag=flag1, criterion='DISCARD_ANY')
    assert mask[0] and not mask[1:].any()  # flag is irrelevant
    mask = flags.flag_mask(test, flag=flag1, criterion='DISCARD_ALL')
    assert not mask[1::2].any() and mask[0::2].all()
    mask = flags.flag_mask(test, flag=flag1, criterion='DISCARD_MATCH')
    assert mask[0] and not mask[1] and mask[2:].all()
    mask = flags.flag_mask(test, flag=flag1, criterion='KEEP_ANY')
    assert not mask[0] and mask[1:].all()  # flag is irrelevant
    mask = flags.flag_mask(test, flag=flag1, criterion='KEEP_ALL')
    assert not mask[0::2].any() and mask[1::2].all()
    mask = flags.flag_mask(test, flag=flag1, criterion='KEEP_MATCH')
    assert not mask[0] and mask[1] and not mask[2:].any()
    with pytest.raises(ValueError) as err:
        _ = flags.flag_mask(test, flag=flag1, criterion='FOO')
    assert 'Invalid criterion flag' in str(err.value)


def test_discard_indices(flags):
    test = np.arange(10)
    inds = flags.discard_indices(test, flag=None, criterion=None)
    assert np.allclose(inds, test[1:])
    inds = flags.discard_indices(test, flag=flag1, criterion='DISCARD_ALL')
    assert np.allclose(inds, np.arange(1, 10, 2))
    test2 = test.reshape((2, 5))
    inds = flags.discard_indices(test2, flag=flag1, criterion='DISCARD_ALL')
    assert np.allclose(inds[0], [0, 0, 1, 1, 1])
    assert np.allclose(inds[1], [1, 3, 0, 2, 4])


def test_flagged_indices(flags):
    test = np.arange(10)
    inds = flags.flagged_indices(test, flag=None, criterion=None)
    assert np.allclose(inds, test[1:])
    inds = flags.flagged_indices(test, flag=flag1, criterion='DISCARD_ALL')
    assert np.allclose(inds, np.arange(0, 10, 2))
    test2 = test.reshape((2, 5))
    inds = flags.flagged_indices(test2, flag=flag1, criterion='DISCARD_ALL')
    assert np.allclose(inds[0], [0, 0, 0, 1, 1])
    assert np.allclose(inds[1], [0, 2, 4, 1, 3])


def test_all_excluding(flags):
    assert flags.all_excluding(flag2) == flag1 | flag3


def test_unflag(flags):
    test_flag = flag1 | flag2
    assert flags.unflag(test_flag, flag3) == test_flag
    assert flags.unflag(test_flag, flag2) == flag1


def test_edit_header(flags):
    for prefix in [None, 'T']:
        header = fits.Header()
        flags.edit_header(header, prefix=prefix)
        p = '' if prefix is None else prefix
        assert len(header) == 3
        assert header[f'{p}FLAG1'] == 'FLAG1'
        assert header[f'{p}FLAG2'] == 'FLAG2'
        assert header[f'{p}FLAG4'] == 'FLAG3'
        assert header.comments[f'{p}FLAG1'] == 'BasicFlagTypes flag 1 (1)'
        assert header.comments[f'{p}FLAG2'] == 'BasicFlagTypes flag 2 (2)'
        assert header.comments[f'{p}FLAG4'] == 'BasicFlagTypes flag 3'

    class NoFlags(Flags):
        flags = None

    header = fits.Header()
    NoFlags.edit_header(header)
    assert len(header) == 0


def test_to_letters(flags):
    assert flags.to_letters(1) == '1'
    letters = flags.to_letters(np.arange(10)).tolist()
    assert letters == ['-', '1', '2', '12', '-', '1', '2', '12', '-', '-']

# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.cartesian_system \
    import CartesianSystem


def test_init():
    c = CartesianSystem(axes=9)
    assert c.size == 9
    names = ['x', 'y', 'z', 'u', 'v', 'w', 't', 't1', 't2']
    for dimension, name in enumerate(names):
        assert c.axes[dimension].label == name

.. currentmodule:: sofia_redux.toolkit.utilities.multiprocessing

`Joblib <https://joblib.readthedocs.io/en/latest/>`_ is a user friendly
package enabling reliable local parallel processing on a wide variety of tasks
and is the recommended method for :mod:`sofia_redux.toolkit`.  Documentation may be
found at `<https://joblib.readthedocs.io>`_.

Usage
-----

:func:`multitask` is the high level wrapper function that makes use of
:mod:`joblib` for processing tasks in parallel.  If :func:`multitask` fails
a second attempt will be made to process the tasks serially.

Multitask Example (Joblib)
--------------------------

    .. code-block:: python

        from sofia_redux.toolkit.utilities.multiprocessing import multitask

        def multi_add_ten(args, i):
            return args[i] + 10

        numbers = [10, 11, 12, 13, 14]
        indices = range(len(numbers))
        multitask(multi_add_ten, indices, numbers, None, jobs=-1)
        # gives [20, 21, 22, 23, 24]

The above code uses :mod:`joblib` to add 10 to a set of numbers in parallel.
`jobs=-1` tells joblib to perform the processing using all available CPUs.
There is virtually no overhead using this method compared with other methods
such as :mod:`dask`.

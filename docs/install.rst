============
Installation
============

Stable release
--------------

The `sofia_redux` package is available via anaconda or pip::

   conda install -c sofia-usra sofia_redux

or::

   pip install sofia_redux


From source
-----------

Via Anaconda
^^^^^^^^^^^^

We recommend Anaconda for managing your Python environment.  A conda
environment specification is included with this package, as `environment.yml`.

To install a ``sofia_redux`` environment with Anaconda::


   conda env create -f environment.yml


Activate the environment::

   conda activate sofia_redux


Install the `sofia_redux` package::

   pip install .


Deactivate the environment when done::

   conda deactivate


Remove the environment if necessary::

   conda remove --name sofia_redux --all


Via Pip
^^^^^^^

Alternately, prerequisites for the package can be installed with::

  pip install -r requirements.txt

and the package can then be installed as usual::

   pip install .

Optional Requirements
---------------------

Some optional visualization tools in the SOFIA Redux interface use the `pyds9`
library to interface with the external SAOImage DS9 tool.  To use these tools,
install `DS9 <https://sites.google.com/cfa.harvard.edu/saoimageds9>`_, then
install pyds9 directly via pip::

  pip install pyds9

or using the provided optional requirements file::

  pip install -r optional_requirements.txt

Please note that this library requires gcc to compile, and is not available
on the Windows platform.

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""GUI interface to DS9 for FITS viewing and analysis."""

if not _ASTROPY_SETUP_:
    from .qad_imview import QADImView
    from .qad_headview import HeaderViewer
    from .qad_app import main as qad_app
    from .qad_main_panel import QADMainWindow
    from .qad_dialogs import PhotSettingsDialog, DispSettingsDialog

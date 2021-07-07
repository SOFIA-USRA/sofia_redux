# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Redux GUI interface and viewers."""

if not _ASTROPY_SETUP_:
    from sofia_redux.pipeline.gui.main \
        import ReduxMainWindow
    from sofia_redux.pipeline.gui.matplotlib_viewer \
        import MatplotlibViewer, MatplotlibPlot
    from sofia_redux.pipeline.gui.qad_viewer \
        import QADViewer, QADViewerSettings
    from sofia_redux.pipeline.gui.textview \
        import TextView
    from sofia_redux.pipeline.gui.widgets \
        import (PipeStep, ProgressFrame, EditParam, DataTableModel,
                TextEditLogger, CustomSignals, RemoveFilesDialog, ParamView,
                StepRunnable, LoadRunnable)


.. _signals:

Signals
=======

Overview
--------

Signals are the most efficient way for GUI applications to communicate
when events occur to other parts of the application. The Eye uses
:class:`pyqtSignal` objects. When the Eye starts up, the signals are
connected to various functions, denoted as a :class:`pyqtSlot` in the control
functions of the Eye. When the signal is activated with the :mod:`emit` method
from anywhere in the Eye, the connected function is called.


Defined
-------

The available signals are defined in the
:class:`Signals <sofia_redux.visualization.signals.Signals>` class. All
connections are configured in the
:class:`Setup <sofia_redux.visualization.setup.Setup>` class.

Used
----


The signals used by the Eye can be classified in a few different categories.

    *  Figure needs to be updated
       -  atrophy
       -  atrophy_bg_partial
       -  atrophy_bg_full

    *  GUI needs to be updated
       -  refresh_order_list
       -  update_pane_tree

    *  GUI was changed
       -  axis_limits_changed
       -  axis_scale_changed
       -  axis_field_changed

    *  A button was pushed
       -  current_pane_changed
       -  display_selected_model
       -  remove_data
       -  clear_selection
       -  end_selection


.. _figure-update-signals:

Figure Updates
^^^^^^^^^^^^^^

Occasionally the figure itself is out of date. This could be for several
reason, such as a new pane is added, the *x*-axis units have changed, or the
cursor moved across the figure. Having the figure update itself constantly
just to catch any possible changes would be inefficient. Instead the Eye uses
the atrophy system, described in detail in :ref:`figure-atrophy`, controlled by
the state of three variables:

*  view.stale
*  view.stale_background_partial
*  view.stale_background_full

If the background of the plot has not be changed (ie all axis parameters are
the same) then only "view.stale" is atrophied. If the background needs to
be updated but the axis scales are still valid, then
"view.stale_background_partial" is atrophied. If the axis scale is no longer
valid, then "view.stale_background_full" is atrophied. When the figure
refresh check next occurs, the stale variables will be checked, and if any
are true, then the figure is updated.


.. _gui-changed-signals:

GUI Has Changed
^^^^^^^^^^^^^^^

One of the ways to customize the figure is by changing the values in the
control fields in the GUI, such as setting the axis limits or changing the
axis scale. When anything in the GUI controls change, the corresponding
signal is emitted, which will result in the correct figure update signal
being emitted as discussed in :ref:`figure-update-signals`. The figure will
then refresh to reflect the current state of the controls.


.. _gui-update-signals:

GUI Updates
^^^^^^^^^^^

The GUI controls are not the only mechanism through with the figure can be
updated. For example, the user could select a range over which the figure
should zoom in to. In that case, the limits displayed in the controls are no
longer accurate. The GUI update signals tell the controls to refresh to match
the current state of the figure. This is the reverse of the signals discussed
in :ref:`gui-changed-signals`.



Button Pressed
^^^^^^^^^^^^^^

The user can control the figure in more ways than through the control panel.
For instance, clicking on a pane will mark it as the current pane. Pressing
the *k* key starts the Gaussian fitting routine. In any of these cases, the
"button pushed" signals interpret the input and calls the correct routines.
The following keyboard signals are implemented:

- **C**: Clear all curve fits and current selections.

- **F**: Start selecting an *x*-axis range to fit a Gaussian to.

- **W**: Reset all axis limit changes to their default values.

- **X**: Start selecting an *x*-axis range to zoom in on.

- **Y**: Start selecting an *y*-axis range to zoom in on.

- **Z**: Start selecting an *x*-axis and *y*-axis range to zoom in on.

- **Return**: Plots the currently selected model in the file table on the
  current axis.

- **Delete**: Remove the currently selected model in the file table or the
  current pane.




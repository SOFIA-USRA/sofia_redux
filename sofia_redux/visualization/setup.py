# Licensed under a 3-clause BSD style license - see LICENSE.rst

from logging import FileHandler

from sofia_redux.visualization import log
from sofia_redux.visualization.utils.logger \
    import StatusLogger, DialogLogger, StreamLogger

__all__ = ['Setup']


class Setup(object):
    """
    Setup GUI controls and callbacks.
    """

    def __init__(self, parent):
        """Initialize GUI setup."""
        self.parent = parent
        self.view = parent.view

    def setup_all(self):
        """Call all setup actions."""
        self.setup_menu_bar()
        self.setup_controls()
        self.setup_line_controls()
        self.setup_model_controls()
        self.setup_mouse_events()
        self.setup_signals()
        self.setup_messages()

    def setup_menu_bar(self):
        """Connect menu bar to callbacks."""
        # menu bar is still TBD
        self.view.menubar.hide()

    def setup_controls(self):
        """Connect control panel events to callbacks."""
        # default: hide a couple control panels
        self.view.order_panel.hide()
        self.view.axis_panel.hide()
        self.view.plot_panel.hide()
        self.view.analysis_panel.hide()

        # order panel has TBD functionality; hide it for now
        self.view.order_panel_frame.hide()

        # connect signals
        self.view.hightlight_pane_checkbox.toggled.connect(
            self.view.toggle_pane_highlight)

        self.view.x_property_selector.activated.connect(self.view.set_field)
        self.view.y_property_selector.activated.connect(self.view.set_field)
        self.view.x_unit_selector.activated.connect(self.view.set_unit)
        self.view.y_unit_selector.activated.connect(self.view.set_unit)
        self.view.x_scale_linear_button.toggled.connect(self.view.set_scale)
        self.view.y_scale_linear_button.toggled.connect(self.view.set_scale)
        self.view.x_scale_log_button.toggled.connect(self.view.set_scale)
        self.view.y_scale_log_button.toggled.connect(self.view.set_scale)
        self.view.x_limit_min.editingFinished.connect(self.view.set_limits)
        self.view.x_limit_max.editingFinished.connect(self.view.set_limits)
        self.view.y_limit_min.editingFinished.connect(self.view.set_limits)
        self.view.y_limit_max.editingFinished.connect(self.view.set_limits)
        self.view.enable_overplot_checkbox.toggled.connect(
            self.view.toggle_overplot)
        self.view.axes_selector.currentIndexChanged.connect(
            self.view.update_controls)

        # zoom controls
        self.view.x_zoom_button.clicked.connect(
            lambda: self.view.start_selection('x_zoom'))
        self.view.y_zoom_button.clicked.connect(
            lambda: self.view.start_selection('y_zoom'))
        self.view.box_zoom_button.clicked.connect(
            lambda: self.view.start_selection('b_zoom'))
        self.view.reset_zoom_button.clicked.connect(
            self.view.reset_zoom)

        # panel collapsing
        self.view.collapse_controls_button.clicked.connect(
            self.view.toggle_controls)
        self.view.collapse_cursor_button.clicked.connect(
            self.view.toggle_cursor)
        self.view.collapse_file_choice_button.clicked.connect(
            self.view.toggle_file_panel)
        self.view.collapse_pane_button.clicked.connect(
            self.view.toggle_pane_panel)
        self.view.collapse_order_button.clicked.connect(
            self.view.toggle_order_panel)
        self.view.collapse_axis_button.clicked.connect(
            self.view.toggle_axis_panel)
        self.view.collapse_plot_button.clicked.connect(
            self.view.toggle_plot_panel)
        self.view.collapse_analysis_button.clicked.connect(
            self.view.toggle_analysis_panel)

        # cursor enabling
        self.view.cursor_checkbox.clicked.connect(
            self.view.enable_cursor_position)
        self.view.cursor_popout_button.clicked.connect(
            self.view.popout_cursor_position)

        # analysis
        self.view.open_fit_results_button.clicked.connect(
            self.view.open_fits_results)

    def setup_line_controls(self):
        """Connect control panel to plot callbacks."""
        # pane controls
        self.view.add_pane_button.clicked.connect(self.view.add_pane)
        self.view.remove_pane_button.clicked.connect(self.view.remove_pane)
        self.view.pane_tree_display.itemChanged.connect(self.view.enable_model)
        self.view.pane_tree_display.itemDoubleClicked.connect(
            self.view.select_pane)

        # plot controls
        self.view.color_cycle_selector.currentTextChanged.connect(
            self.view.select_color_cycle)
        self.view.plot_type_selector.currentTextChanged.connect(
            self.view.select_plot_type)
        self.view.marker_checkbox.toggled.connect(
            self.view.toggle_markers)
        self.view.grid_checkbox.toggled.connect(
            self.view.toggle_grid)
        self.view.error_checkbox.toggled.connect(
            self.view.toggle_error)
        self.view.dark_mode_checkbox.toggled.connect(
            self.view.toggle_dark_mode)

    def setup_model_controls(self):
        """Connect control panel to model callbacks."""
        self.view.add_file_button.clicked.connect(self.parent.add_data)
        self.view.remove_file_button.clicked.connect(self.parent.remove_data)

    def setup_mouse_events(self):
        """Connect mouse events to callbacks."""
        self.view.file_table_widget.itemDoubleClicked.connect(
            self.parent.display_selected_model)
        self.view.figure_widget.canvas.mpl_connect(
            'button_press_event', self.view.figure_clicked)

    def setup_signals(self):
        """Connect signals to callbacks."""
        self.view.signals.atrophy.connect(self.view.atrophy)
        self.view.signals.atrophy_controls.connect(self.view.atrophy_controls)
        self.view.signals.atrophy_bg_full.connect(
            self.view.atrophy_background_full)
        self.view.signals.atrophy_bg_partial.connect(
            self.view.atrophy_background_partial)
        self.view.signals.refresh_order_list.connect(
            self.view.refresh_order_list)

        self.view.signals.axis_limits_changed.connect(
            self.view.axis_limits_changed)
        self.view.signals.axis_scale_changed.connect(
            self.view.axis_scale_changed)
        self.view.signals.axis_unit_changed.connect(
            self.view.axis_unit_changed)
        self.view.signals.axis_field_changed.connect(
            self.view.axis_field_changed)

        self.view.signals.panes_changed.connect(self.view.update_pane_tree)
        self.view.signals.current_pane_changed.connect(
            self.view.current_pane_changed)
        self.view.signals.model_selected.connect(
            self.parent.display_selected_model)
        self.view.signals.model_removed.connect(
            self.parent.remove_data)
        self.view.signals.end_zoom_mode.connect(
            self.view.clear_selection)
        self.view.signals.end_cursor_recording.connect(
            self.view.end_selection)
        self.view.signals.clear_fit.connect(self.view.clear_fit)
        self.view.signals.toggle_fit_visibility.connect(
            self.view.toggle_fit_visibility)

    def setup_messages(self):
        """Add filters to pass log messages to GUI handlers."""

        # make sure terminal matches parent setting;
        # remove any old GUI or file handlers
        for hand in log.handlers:
            if isinstance(hand, StreamLogger):
                hand.setLevel(self.parent.log_level)
            elif isinstance(hand, StatusLogger) \
                    or isinstance(hand, DialogLogger) \
                    or isinstance(hand, FileHandler):
                log.removeHandler(hand)

        # INFO goes to status bar
        logger = StatusLogger(self.view.statusbar)
        logger.setLevel('INFO')
        log.addHandler(logger)

        # WARNING/ERROR goes to dialog box
        logger = DialogLogger(self.view)
        logger.setLevel('WARNING')
        log.addHandler(logger)

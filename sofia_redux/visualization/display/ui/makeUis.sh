#!/usr/bin/env bash

#
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#

pyuic5 simple_spec_viewer.ui -o simple_spec_viewer.py
pyuic5 fit_result_window.ui -o fit_result_window.py
pyuic5 cursor_location.ui -o cursor_location.py


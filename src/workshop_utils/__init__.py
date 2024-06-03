#!/usr/bin/env python3

import importlib.resources
from . import plotting
from .fetch import fetch_data, DOWNLOADABLE_FILES
STYLE_FILE = importlib.resources.files('workshop_utils') / 'nemos.mplstyle'

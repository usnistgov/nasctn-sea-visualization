# -----------------------------------------------------------------------------
# Authors:     aric.sanders@nist.gov
# Created:     07/21/2025
# License:     NIST License
# -----------------------------------------------------------------------------
"""
This package is a set of tools for data reshaping and visualization of the
 sensor data from the NACSTN CBRS SEA program.

"""

#----------------------------------------------------------------------
# Standard Imports
import os
import sys

# -----------------------------------------------------------------------------
# Third Party Imports
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..'))

# -----------------------------------------------------------------------------
# Module Constants
VERBOSE_IMPORT = False
TIMED_IMPORT = False
__version__ = "0.1.2"
"Constant that determines if import statements are echoed to output"
# The new module load scheme can be for module in DE_API_MODULES.keys()
# -----------------------------------------------------------------------------
DE_API_MODULES = {"nasctn_sea_visualization.data_models":True,
                  "nasctn_sea_visualization.plotters":True,
                  "nasctn_sea_visualization.reports":True}
"Dictionary that controls the definition of the API, this can be set to leave out any unwanted modules. Also it is" \
    "possible to discover all modules by DE_API_MODULES.keys()"

# This makes sure this file is the one loaded
sys.path.append(os.path.dirname(__file__))
# To tune the imported API change the DE_API_MODULES dictionary
if TIMED_IMPORT:
    import datetime

    first_timer = datetime.datetime.now(datetime.timezone.utc)
    start_timer = datetime.datetime.now(datetime.timezone.utc)
for module in sorted(DE_API_MODULES.keys()):
    try:
        if DE_API_MODULES[module]:
            if VERBOSE_IMPORT:
                print(("Importing {0}".format(module)))
            exec('from {0} import *'.format(module))
            if TIMED_IMPORT:
                end_timer = datetime.datetime.now(datetime.timezone.utc)
                time_difference = end_timer - start_timer
                print(("It took {0} s to import {1}".format(time_difference.total_seconds(), module)))
                start_timer = end_timer
    except:
        print(f"The {module}failed to import")
        pass
if TIMED_IMPORT:
    end_timer = datetime.datetime.now(datetime.timezone.utc)
    time_difference = end_timer - first_timer
    print(("It took {0} s to import all of the active modules".format(time_difference.total_seconds())))
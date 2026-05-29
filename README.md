# nasctn-sea-visualization
Tools to open, reshape and visualize data from the NASCTN CBRS SEA program.

# Installation 
## Method 1 - Directly from Github
```shell
pip install git+https://github.com/usnistgov/nasctn-sea-visualization.git 
```
## Method 2 - Clone then Install
1. Use your favorite way to clone the repo or download the zip file and extract it.  
2. pip install to that location (note pip expects a unix style string 'C:/univariate_tools/')

```shell
pip install <path to top folder>
```
# Use case
Once you have access to the NASCTN CBRS SEA data, you need plots or tables for analysis purposes. 
# Workflow
1. Access the [NASCTN CBRS SEA Data](https://pages.nist.gov/SEA-DATA/) 
2. Typically a 


## Example use
```python
import sys 
import os
sys.path.append(r"C:\Users\sandersa\VSCode Repos\sea_tier2_addons") #this should point to your local installation directory
from data_models import *
from plotters import *
``` 

## Common Plots
```python
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r"C:\Users\sandersa\VSCode Repos\sea_tier2_addons") #this should point to your local installation directory
from data_models import *
from plotters import *
hu_path = r".\Raw Data\2024-08-10_HU.zip"
hu_day_block = DayBlock(file_path = hu_path)
hu_plotter = DayBlockPlotter(hu_day_block)
# This plots mean psd for the full day
hu_plotter.plot_day_psd(capture_statistic="mean")
#this plots the day plot
hu_plotter.plot_day()
# This plots the aligned PFP for the 11 channel, you can use frequency as the selector also
hu_plotter.plot_aligned_pfp(channel=11)
# This plots a DL /UL separation attempt
hu_plotter.plot_day_link_separation(channel=11)
# This selects a single sigmf, slice can be in a pd.datetime object too
hu_slice = DataProduct(data=hu_day_block.get_single(100))
hu_slice_plotter = DataProductPlotter(hu_slice)
# plots a channel pfp and histogram 
hu_slice_plotter.plot_channel_pfp_hist(11)
# plots a channel summary
hu_slice_plotter.plot_channel_summary(hu_day_block.frequencies[10])
#plots all psd's for a single slice
hu_slice_plotter.plot_all_psds()
#plots a single channel pfp
hu_slice_plotter.plot_channel_pfp(hu_day_block.frequencies[0])
```

# Example
An [example](./examples/Example.ipynb) of opening data, reshaping it to tables of interest and plotting common elements.

# API Documentation
The [API Documentation](https://pages.nist.gov/nasctn-sea-visualization) links to the __init__.py file and has the primary submodules linked. 

# Contact
Aric Sanders [aric.sanders@nist.gov](mailto:aric.sanders@nist.gov)


# NIST Disclaimer
Certain commercial equipment, instruments, or materials (or suppliers, or software, ...) are identified in this repository to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.


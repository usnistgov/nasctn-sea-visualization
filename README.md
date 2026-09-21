# nasctn-sea-visualization
Tools to open, reshape and visualize data from the NASCTN CBRS SEA program.

# Installation 
## Method 1 - Directly from Github
```shell
pip install git+https://github.com/usnistgov/nasctn-sea-visualization.git 
```
## Method 2 - Clone then Install
1. Use your favorite way to clone the repo or download the zip file and extract it.  
2. pip install to that location (note pip expects a unix style string 'C:/nasctn-sea-visualization/')

```shell
pip install <path to top folder>
```
# Environment Setup 
This package is sensitive to the requirements for [nasctn-sea-ingest](https://github.com/usnistgov/nasctn-sea-ingest) and needs older versions of numpy and pandas to operate properly. It is recommended that a virtual environment is used with python 3.9.23 -
<hr/>
An example batch file to load the requirements:

```batch 
call "%LOCALAPPDATA%\miniforge3\Scripts\activate.bat" 
call conda create -n sea python=3.9.23
call conda activate sea
call pip install git+https://github.com/usnistgov/nasctn-sea-ingest
call pip install seaborn==0.12.0
call pip install hvplot==0.8.0
call pip install openpyxl
call pip install scipy==1.11.0
```
# Use case
Once you have access to the NASCTN CBRS SEA data, you need plots or tables for analysis purposes. 

# Workflow
1. Access the [NASCTN CBRS SEA Data](https://pages.nist.gov/SEA-DATA/) , for a single day you can use this code snippet to download data:
```python
import requests

url = "https://data.nist.gov/od/ds/mds2-4214/Raw%20Data/2024-07-15_GMM.zip"
local_filename = "2024-07-15_GMM.zip"

# Send a GET request to the URL
response = requests.get(url)

# Open a local file in write-binary mode and save the content
with open(local_filename, "wb") as file:
    file.write(response.content)
```


2. Choose the date and sensor of interest
3. If the desired data is in csv format (PSD, PFP, Summary) plot directly from there (data footprint is much smaller)
4. If in-depth analysis is required for the (day, sensor) combination, load the data from Raw Data directory using the DayBlock class. This class has several different methods for producing flat tables and other data slices. 
5. If plots are required create a DayBlockPlotter object by passing it the DayBlock object from 4.
6. Plot  


## Example use
```python
import sys 
import os
from nasctn_sea_visualization.data_models import *

``` 

## Common Plots
```python
import sys 
import os
import numpy as np
import matplotlib.pyplot as plt

from nasctn_sea_visualization import *
hu_path = r".\Raw Data\2024-08-10_HU.zip"
hu_day_block = DayBlock(file_path = hu_path)
hu_plotter = DayBlockPlotter(hu_day_block)
# This plots mean psd for the full day
hu_plotter.plot_day_psd(capture_statistic="mean")
#this plots the day plothu_plotter.plot_day()
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
For a general example see [example](./examples/Example.ipynb), for a specific example of opening data, reshaping it to tables of interest see [Example of Getting Data Products](./examples/Example_of_Getting_Data_Products.ipynb). For an example of comparing percentiles of power spectral density see [Comparison of PSDs](./examples/Example_Comparing_PSDs.ipynb). An example of splitting CBSD activity into uplink and downlink is at [Example of DL/UL Splitting](./examples/Example_DL_UL_splitting.ipynb)

# API Documentation
The [API Documentation](https://pages.nist.gov/nasctn-sea-visualization) links to the __init__.py file and has the primary submodules linked.  

# Contact
Aric Sanders [aric.sanders@nist.gov](mailto:aric.sanders@nist.gov)


# NIST Disclaimer
Certain commercial equipment, instruments, or materials (or suppliers, or software, ...) are identified in this repository to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.


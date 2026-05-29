# -----------------------------------------------------------------------------
# Name:        
# Purpose:    
# Authors:     aric.sanders@nist.gov
# Created:     
# License:     NIST License
# -----------------------------------------------------------------------------
"""plotters.py is a module designed to encapsulate a data_model from data_models.py and provide visualizations for 
"""
# -----------------------------------------------------------------------------
# Standard Imports
import numpy as np
import lzma
import os 
#import numexpr as ne
import tarfile
import json
import re
import datetime
import dateutil as dtu
import datetime as dt
import sys
# -----------------------------------------------------------------------------
# Third Party Imports
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib import ticker
import sea_ingest
import seaborn as sns
import hvplot.pandas
from data_models import * #replace with absolute import
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname( __file__ ),'.'))

# -----------------------------------------------------------------------------
# Module Constants
# -----------------------------------------------------------------------------
# Module Functions
def summary_classifier(MMPR,PMPR,
                              cbrs_slope=1.25,cbrs_intercept=18.5,cbrs_tolerance=7,cbrs_min_mmpr=2.25,
                              noise_mmpr_min=1,noise_mmpr_max=2.5,noise_pmpr_min =13,noise_pmpr_max = 35):
    """Uses the peak to median power ratio and mean to median power ratio to determine 
    category. Category 0 - CBSD, 1- Noise, 2- Signal. The CBRS model is a simple linear model 
    (cbrs_slope*mmpr+cbrs_intercept) with a tolerance. The Noise model is a simple check to see if the value is within the rectangular region
    made by noise_mmpr_min,noise_mmpr_max,noise_pmpr_min,and noise_pmpr_max. """
    if (MMPR>cbrs_min_mmpr)&(np.abs(PMPR-(cbrs_slope*MMPR+cbrs_intercept))<cbrs_tolerance):
        return 0
    elif (noise_mmpr_min<MMPR<noise_mmpr_max)&(noise_pmpr_min<PMPR<noise_pmpr_max):
        return 1
    else:
        return 2 
classifier_  = np.vectorize(summary_classifier)

def dbsum(x):
    y = np.sum(10 ** (x / 10))
    ydb = 10 * np.log10(y)
    return ydb


def dbmean(x):
    y = np.mean(10 ** (x / 10))
    ydb = 10 * np.log10(y)
    return ydb


def resampledf(psd):
    xx = psd.T
    idx = xx.index
    idx = pd.DataFrame(idx.groupby(np.arange(len(xx.index)) // 25)).mean(axis=0)
    xx = xx.groupby(np.arange(len(xx.index)) // 25, axis=0).apply(dbmean)
    xx.index = idx.values
    return xx.T


def limiter(df, index_key="frequency"):
    df_g = df.reset_index(level="datetime").groupby(level=[index_key])
    lim = df_g.size().min()
    df_lim = df_g.nth[:lim]
    df_lim = df_lim.reset_index().set_index([index_key, "datetime"])
    return df_lim

def day_psd(day_df):
    psd = resampledf(
        day_df["psd"]
    )  # resamples the PSD to a more meaningful number of values for plotting.
    df = (
        psd.xs("max", level="capture_statistic")
        .swaplevel("frequency", "datetime")
        .sort_index()
    )

    df_lim = limiter(
        df
    )  # limiter is a coarse method to adjust limitations due to un-equal number of per frequency captures during that day
    f0 = df_lim.index.get_level_values(
        "frequency"
    ).unique()  # core frequency components for the colormaps
    dfc = df_lim.xs(
        f0.min(axis=0), level="frequency"
    )  # sets the datetimestamp of the data to that of the lowest frequency
    timestamps = dfc.index
    basef = dfc.columns.astype(float)
    freqs = np.array(
        [np.array(basef) + c for c in np.array(f0)]
    ).flatten()  # combines the channel frequecny with the base frequency within the channel
    dfr = (
        df_lim.reset_index("datetime", drop=True)
        .unstack("frequency")
        .T.swaplevel()
        .sort_index()
    )

    X, Y = np.meshgrid(timestamps, freqs / 1e6)
    psd_image = {"time": X.T, "freq": Y.T, "psd": np.reshape(dfr.values, X.shape).T}
    return psd_image

def day_overload(day_df):
    ovr = pd.DataFrame(day_df["channel_metadata"]["overload"]).swaplevel().sort_index()
    ovr = ovr.overload.map({True: -1, False: 1})
    ovr_lim = limiter(ovr)
    f0 = ovr_lim.index.get_level_values("frequency").unique()
    ovr_c = ovr_lim.xs(f0.min(), level="frequency")
    timestamps = ovr_c.index

    X, Y = np.meshgrid(timestamps, f0 / 1e6)
    ovr_image = {"time": X.T, "freq": Y.T, "ovr": np.reshape(ovr_lim.values, X.shape).T}
    return ovr_image

def day_pfp(day_df):
    pfp_median = (
        day_df["pfp"]
        .xs(["mean", "rms"], level=["capture_statistic", "detector"], drop_level=True)
        .median(axis=1)
    )
    pfp_max = (
        day_df["pfp"]
        .xs(["max", "peak"], level=["capture_statistic", "detector"], drop_level=True)
        .max(axis=1)
    )
    pfp = pd.concat([pfp_median, pfp_max], axis=1)
    pfp.columns = ["median", "max"]
    pfp = pfp.swaplevel().sort_index()
    pfp_lim = limiter(pfp)

    f0 = pfp_lim.index.get_level_values("frequency").unique()
    pfp_c = pfp_lim.xs(f0.min(axis=0), level="frequency")
    timestamps = pfp_c.index

    X, Y = np.meshgrid(timestamps, f0 / 1e6)
    pfp_image = {
        "time": X.T,
        "freq": Y.T,
        "pfp_median": np.reshape(pfp_lim["median"].values, X.shape).T,
        "pfp_max": np.reshape(pfp_lim["max"].values, X.shape).T,
    }
    return pfp_image

def detector_plot(
    day_df, sensname, savename_pass=False, watermark=False, tz_info="UTC",annotation_date = False,show=False,
):
    psd_image = day_psd(day_df)
    pfp_image = day_pfp(day_df)
    ovr_image = day_overload(day_df)
    doi_str = ovr_image["time"][5,5].date()

    fig, axs = plt.subplots(
        1,
        4,
        sharey=True,
        figsize=(15, 10),
    )

    # Overload data
    cmap_w = colors.ListedColormap(
        ["pink", "white"]
    )  # custom colormap for Overload conditions mapped to the boolean

    pc0 = axs[0].pcolormesh(
        ovr_image["freq"],
        ovr_image["time"],
        ovr_image["ovr"],
        cmap=cmap_w,
        vmin=-1,
        vmax=1,
    )
    c0 = fig.colorbar(pc0, ax=axs[0], location="top", label="Overload")

    c0.ax.set_xticks([-0.5, 0.5])  # colormap tick adjustement for position and label
    c0.ax.set_xticklabels(["True", "False"])

    # adjustment to make the datetime stamps work out for the y-axis and contain the entire day for the specified timezone
    y_start = pd.to_datetime(doi_str).tz_localize(tz_info)
    axs[0].set_ylim(y_start, y_start + dt.timedelta(hours=24))
    a = axs[0].get_ylim()
    b = axs[0].set_yticks(np.linspace(a[0], a[-1], 25))  # hourly tick marks
    axs[0].yaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=dtu.tz.gettz(tz_info))
    )  # formats y-axis to Hours:Minutes
    axs[0].set_ylabel(f"{y_start} {tz_info}")

    # PSD data
    index = 3  # hacky way of changing order in the plots
    pc2 = axs[index].pcolormesh(
        psd_image["freq"],
        psd_image["time"],
        psd_image["psd"],
        cmap="Greys_r",
        vmin=-155,
        vmax=-115,
    )
    fig.colorbar(
        pc2,
        ax=axs[index],
        shrink=1,
        location="top",
        label="PSD (dBm/Hz)",
        extend="both",
    )

    # PFP Mean RMS
    index = 1
    pc3 = axs[index].pcolormesh(
        pfp_image["freq"],
        pfp_image["time"],
        pfp_image["pfp_median"],
        cmap="summer",
        vmin=-100,
        vmax=-65,
    )
    fig.colorbar(
        pc3,
        ax=axs[index],
        shrink=1,
        location="top",
        label="mean rms pfp (dBm/10MHz)",
        extend="max",
    )
    # PFP Max Peak (or Max Max)
    index = 2
    pc4 = axs[index].pcolormesh(
        pfp_image["freq"],
        pfp_image["time"],
        pfp_image["pfp_max"],
        cmap="spring",
        vmin=-85,
        vmax=-50,
    )
    fig.colorbar(
        pc4,
        ax=axs[index],
        shrink=1,
        location="top",
        label="max peak pfp (dBm/10MHz)",
        extend="both",
    )

    # global title of the figure
    fig.suptitle(f"{sensname}\n Date: {doi_str}", fontsize=15)

    # sets the channel view

    for ax in axs:
        ax.grid(axis="y")
        ax.set_xlim([3540, 3660])
        ax.set_xlabel("Frequency (MHz)")
        if annotation_date:
            ax.axhline(pd.to_datetime(annotation_date),color ="k",linewidth=3,linestyle="dashed")

        # defines Watermarks for the images for draft data requirements
        if watermark:
            ax.text(
                0.5,
                0.5,
                "NASCTN Draft Data",
                transform=ax.transAxes,
                fontsize=15,
                color="gray",
                alpha=0.5,
                ha="center",
                va="center",
                rotation=30,
            )
            ax.text(
                0.5,
                0.15,
                "NASCTN Draft Data",
                transform=ax.transAxes,
                fontsize=15,
                color="gray",
                alpha=0.5,
                ha="center",
                va="center",
                rotation=30,
            )
            ax.text(
                0.5,
                0.85,
                "NASCTN Draft Data",
                transform=ax.transAxes,
                fontsize=15,
                color="gray",
                alpha=0.5,
                ha="center",
                va="center",
                rotation=30,
            )
    
    # defines the saving parameter
    if show:
        plt.show()
    if savename_pass:
        figurename = os.path.join(
            savename_pass, f"Day_Activity_inspect_{sensname}_doi_{doi_str}.png"
        )
        plt.savefig(figurename)
        plt.close()
        print(f"saved {figurename}")

def day_pfp_plot(pfp_roll,freq,sensname,savename_pass=False,watermark=False,tz_info='UTC',tstart_shift = None,hrs =24):
    reserve_re =[3*7+3,4*7-2,13*7+3,14*7-2] #Config 2 - Special 7 - 4 Antennas, 10 MHz, Reserve Elements Start,End, Start, End
    ul_re = [4*7-2,6*7,14*7-2,8*14] ##Config 2 - Special 7 - 4 Antennas, 10 MHz, Uplink Elements Start, End,Start, End
    doi_str = pfp_roll.index.min().date()
    
    fig,axs = plt.subplots(2,1,sharex=True,figsize=(8,7),gridspec_kw={'height_ratios': [4, 1]})

    X,Y = np.meshgrid(pfp_roll.index,pfp_roll.columns.values.astype('float'))
    droll_2dat = np.reshape(pfp_roll.values,X.T.shape).T

    roll_stats = pfp_roll.describe(percentiles=[0.025,0.25,0.5,0.75,0.975]).iloc[3:]

    axs[0].pcolormesh(Y,X,droll_2dat,cmap='viridis')
    for i in reserve_re:
        val = 4*i*0.01/560
        axs[0].axvline(val,color='m')

    for i in ul_re:
        val = 4*i*0.01/560
        axs[0].axvline(val,color='r')

        
    y_start = pd.to_datetime(doi_str).tz_localize(tz_info)
    if tstart_shift!=None:
        y_start = y_start+dt.timedelta(hours=tstart_shift)
    axs[0].set_ylim(y_start,y_start+dt.timedelta(hours=hrs))
    a= axs[0].get_ylim()
    b = axs[0].set_yticks(np.linspace(a[0],a[-1],hrs+1)) #hourly tick marks
    
    axs[0].yaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=dtu.tz.gettz(tz_info)))
    axs[0].set_ylabel(f'{doi_str} {tz_info}, Hour:Minutes')
    axs[0].set_title(f'Sensor: {sensname}, Date: {doi_str} \n Frequency: {freq/1e6} MHz\n PFP aligned to REs using Pearsons correlation coefficients')



    axs[1].plot(pfp_roll.columns.values,roll_stats.T)
    axs[1].legend(roll_stats.index,ncol=len(roll_stats.index),loc=3)
    axs[1].set_ylabel('Power (dBm/10 MHz)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Explicit Percentiles')
    
    for ax in axs:
    #defines Watermarks for the images for draft data requirements
        if watermark:  
            ax.text(0.5, 0.5, 'NASCTN Draft Data', transform=ax.transAxes,
            fontsize=15, color='gray', alpha=0.75,
            ha='center', va='center', rotation=30)
            ax.text(0.15, 0.5, 'NASCTN Draft Data', transform=ax.transAxes,
                        fontsize=15, color='gray', alpha=0.75,
                        ha='center', va='center', rotation=30)
            ax.text(0.85, 0.5, 'NASCTN Draft Data', transform=ax.transAxes,
                        fontsize=15, color='gray', alpha=0.75,
                        ha='center', va='center', rotation=30)
    plt.tight_layout()
    return fig

def plot_psd_csv(file_path):
    """Plots the power spectral density saved as a .csv in the PSD directory"""
    psd_df = pd.read_csv(file_path,index_col=0)
    psd_df.index = pd.to_datetime(psd_df.index)
    colormesh = plt.pcolormesh(1e-6*psd_df.columns.astype(float),psd_df.index,psd_df)
    ax= plt.gca()
    figure = plt.gcf()
    figure.colorbar(colormesh,ax=ax,label="dBm/Hz")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (UTC)")
    plt.show()

def plot_pfp_csv(file_path,frequency=3605e6):
    """Plots the periodic frame power saved as a .csv in the PFP directory,
      with the option to select a specific frequency"""
    pfp_df= pd.read_csv(file_path,index_col=0)
    pfp_df.index = pd.to_datetime(pfp_df.index)
    #data selection
    x_data = np.linspace(0,10,560)
    selected_data = pfp_df[pfp_df["frequency"]==frequency]
    selected_data = selected_data.drop("frequency",axis=1)
    #plotting
    fig, ax = plt.subplots()
    colormesh = ax.pcolormesh(x_data,selected_data.index,selected_data,cmap="viridis")
    ax.set_ylabel("Time (UTC)")
    ax.set_xlabel("Periodic Frame Time (ms) ")
    fig.colorbar(colormesh,ax=ax,label="dBm/10 MHz")
    plt.show()

def plot_summary_csv(file_path,sensor="HU",stream="max"):
    """Plots the summary data saved as a .csv in the Summaries directory,
      with the option to select a sensor and a stream (max,mean, or median)"""
    summary = pd.read_csv(file_path)
    summary["acquisition_timestamp"] = pd.to_datetime(summary["acquisition_timestamp"])
    selected_data=summary[summary["sensor_name"]==sensor]
    pivot = selected_data.pivot(index="acquisition_timestamp",
    columns="channel_frequency_mhz", values=stream)
    colormesh=plt.pcolormesh(pivot.columns,pivot.index,pivot.values,
    cmap="viridis",rasterized=True)
    plt.ylabel("Time (UTC)")
    plt.xlabel("Frequency (MHz)")
    figure = plt.gcf()
    figure.colorbar(colormesh,label="dBm/10 MHz")
    file_name = os.path.basename(file_path)
    plt.title(f"{sensor} for {file_name.replace('.csv','')}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Module Classes

class DataProductPlotter():
    def __init__(self,data: DataProduct) -> None:
        self.data_product = data
        
    def plot_channel_psds(self,channel,capture_statistics='all',**options):
        """Plots all of the capture statistics for the selected channel"""
        defaults = {'max':{"color":"red","alpha":.5,"linestyle":"solid","linewidth":"2","label":'max'},
                    'mean':{"color":"b","alpha":.5,"linestyle":"dashed","linewidth":"2","label":'mean'},
                    'median':{"color":"k","alpha":.5,"linestyle":"dashed","linewidth":"2","label":'median'},
                    '25th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'25th_percentile'},
                    '75th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'75th_percentile'},
                    '90th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'90th_percentile'},
                    '95th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'95th_percentile'},
                    '99th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99th_percentile'},
                    '99.9th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99.9th_percentile'},
                    '99.99th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99.99th_percentile'},
                    "legend":True,
                    "save":False,
                    "show":True,
                    "grid":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value
        if isinstance(capture_statistics,str) and re.search('all',capture_statistics,re.IGNORECASE):
            capture_statistics = self.data_product.psd_statistics
        figure,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,6))
        frequency = self.data_product.get_channel_index(channel)[1]
        for capture_statistic in capture_statistics:
            x,y = self.data_product.get_xy_data(data_product='psd',channel = channel,capture_statistic=capture_statistic)
            plt.plot((x+frequency)/1e6,y,**plot_options[f"{capture_statistic}"])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power Spectral Density (dBm/Hz)")
        if plot_options["grid"]:
            plt.grid()
        if plot_options['legend']:
            plt.legend()
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()

    def plot_all_psds(self,capture_statistics='all',**options):
        """Plots the selected capture statistics for the data product"""
        defaults = {'max':{"color":"red","alpha":.5,"linestyle":"solid","linewidth":"2","label":'max'},
                    'mean':{"color":"b","alpha":.5,"linestyle":"dashed","linewidth":"2","label":'mean'},
                    'median':{"color":"k","alpha":.5,"linestyle":"dashed","linewidth":"2","label":'median'},
                    '25th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'25th_percentile'},
                    '75th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'75th_percentile'},
                    '90th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'90th_percentile'},
                    '95th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'95th_percentile'},
                    '99th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99th_percentile'},
                    '99.9th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99.9th_percentile'},
                    '99.99th_percentile':{"color":"k","alpha":.25,"linestyle":"dashed","linewidth":"2","label":'99.99th_percentile'},
                    "legend":True,
                    "save":False,
                    "show":True,
                    "grid":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value
        if isinstance(capture_statistics,str) and re.search('all',capture_statistics,re.IGNORECASE):
            capture_statistics = self.data_product.psd_statistics

        figure,ax = plt.subplots(nrows=1,ncols=1,figsize = (10,6))
        for frequency in self.data_product.frequencies:
            for capture_statistic in capture_statistics:
                x,y = self.data_product.get_xy_data(data_product='psd',channel = frequency,capture_statistic=capture_statistic)
                plt.plot((x+frequency)/1e6,y,**plot_options[f"{capture_statistic}"])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power Spectral Density (dBm/Hz)")
        if plot_options["grid"]:
            plt.grid()
        if plot_options['legend']:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()  

    def plot_channel_pfp(self,channel,**options):
        defaults = {"legend":True,
                    "save":False,
                    "show":True,
                    "grid":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value  
        x,peak_max = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="max",detector="peak")
        x,peak_mean = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="mean",detector="peak")              
        x,peak_min = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="min",detector="peak")              
        x,rms_max = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="max",detector="rms")
        x,rms_mean = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="mean",detector="rms")              
        x,rms_min = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="min",detector="rms")   
        plt.figure(figsize=(10,6))
        plt.plot(x,peak_mean,color='r',alpha=1,linestyle="dashed",label="Peak")
        plt.fill_between(x,peak_min,peak_max,color='r',alpha=.25)   
        plt.plot(x,rms_mean,color='b',alpha=1,linestyle="dashed",label="RMS")
        plt.fill_between(x,rms_min,rms_max,color='b',alpha=.25)  
        plt.xlabel("Time (s)")
        plt.ylabel("Power (dBm/10 MHz)")       
        if plot_options["grid"]:
            plt.grid()
        if plot_options['legend']:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()  

    def plot_channel_pfp_hist(self,channel,**options):
        defaults = {"legend":True,
                    "save":False,
                    "show":True,
                    "grid":True,
                    "capture_statistic":"mean",
                    "detector":"rms"}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value  
        
        x_data,y_data = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic=plot_options["capture_statistic"],detector=plot_options["detector"])
        #figure, ax = plt.subplots(figsize=(10,6))
        sns.displot(y= y_data,kind='hist',alpha=.1,bins= 100,stat='density',height=6, aspect=2)
        ax=plt.gca()
        ax.set_ylabel("Power (dBm/10 MHz)")
        twin_ax = ax.twiny()
        twin_ax.plot(x_data,y_data,"r")
        twin_ax.set_xlabel("Periodic Time (s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Power (dBm/10 MHz)")  
        plt.tight_layout()     
        if plot_options["grid"]:
            plt.grid()
        if plot_options['legend']:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()
        figure = plt.gcf()
        return figure

    def plot_channel_pvt(self,channel,**options):
        defaults = {"legend":True,
                    "save":False,
                    "show":True,
                    "grid":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value  
        x,pvt_max = self.data_product.get_xy_data(data_product='pvt',channel = channel,detector="maximum")
        x,pvt_rms = self.data_product.get_xy_data(data_product='pvt',channel = channel,detector="rms")
 
        plt.figure(figsize=(10,6))
        plt.plot(x,pvt_max,color='r',alpha=1,linestyle="solid",label="Maximum")
        plt.plot(x,pvt_rms,color='b',alpha=1,linestyle="solid",label="Mean")
        plt.xlabel("Time (s)")
        plt.ylabel("Power (dBm/10 MHz)")       
        if plot_options["grid"]:
            plt.grid()
        if plot_options['legend']:
            plt.legend()
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show() 

    def plot_channel_apd(self,channel,**options):
            """Plots the apd for the selected capture, provide the capture_id as a dictionary"""
            defaults = {"legend":True,
                        "save":False,
                        "show":True,
                        "grid":True}
            plot_options = {}
            for key,value in defaults.items():
                plot_options[key] = value
            for key,value in options.items():
                plot_options[key] = value
            x,y = self.data_product.get_xy_data(data_product='apd',channel = channel)
            # one solution is to create a new series, swapping the axis for the channel data:
            apd = pd.Series(y,index = x)
            # plot setup: define x-axis ticks
            # note: this configuration results in some data not being displayed
            # at the highest and lowest probabilities. Adjust here or in
            # the "ax.set_xlim" call to change this, if desired.
            xtick_labels = [
                "1e-4", "1e-2", "1e-1", "1", "5", "10", "20", "30", "40",
                "50", "60", "70", "80", "90", "95", "98", "99", #"99.9",
            ]
            ptick_values = np.array([float(x) / 100. for x in xtick_labels])
            x_origin = 10.0 * np.log10(-np.log(ptick_values[0]))
            # Map p value ticks to x values
            xtick = x_origin - 10.0 * np.log10(-np.log(ptick_values))
            # Map p values to x axis, dealing with NaNs
            p = np.where(apd.index != 0, apd.index, np.nan)
            logp = np.where(np.log(p) != 0, np.log(p), np.nan)
            x = x_origin - 10.0 * np.log10(-logp)
            # plot
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(x, apd.values)
            ax.set_xlabel("Percent Exceeding Ordinate")
            ax.set_xticks(xtick, xtick_labels)
            ax.set_xlim(xtick.min(), xtick.max())
            ax.set_ylabel("Channel Power (dBm / 10 MHz)")
            if plot_options["grid"]:
                ax.grid(True, "minor", "y", alpha=0.75)
                ax.grid(True, "major", "both", alpha=0.5)
            if plot_options['legend']:
                plt.legend()
            if plot_options["save"]:
                plt.savefig(plot_options["save"])
            if plot_options["show"]:
                plt.show()           

    def plot_channel_summary(self,channel,**options):
            """Plots all data products for the selected channel, with an overview psd"""
            defaults = {"legend":True,
                        "save":False,
                        "show":True,
                        "grid":True,
                        "figsize":(12,6),
                        "axes_labels":True,
                        "plot_titles":True}
            plot_options = {}
            for key,value in defaults.items():
                plot_options[key] = value
            for key,value in options.items():
                plot_options[key] = value
            channel_frequency = self.data_product.get_channel_index(channel)[-1]
            
            #######################################################################
            # Setup APD Plot
            x,y = self.data_product.get_xy_data(data_product='apd',channel = channel)
            # one solution is to create a new series, swapping the axis for the channel data:
            apd = pd.Series(y,index = x)
            # plot setup: define x-axis ticks
            # note: this configuration results in some data not being displayed
            # at the highest and lowest probabilities. Adjust here or in
            # the "ax.set_xlim" call to change this, if desired.
            xtick_labels = [
                "1e-4", "1", "5", "20",
                "50","75","90", "95", "98", "99", #"99.9",
            ]
            ptick_values = np.array([float(x) / 100. for x in xtick_labels])
            x_origin = 10.0 * np.log10(-np.log(ptick_values[0]))
            # Map p value ticks to x values
            xtick = x_origin - 10.0 * np.log10(-np.log(ptick_values))
            # Map p values to x axis, dealing with NaNs
            p = np.where(apd.index != 0, apd.index, np.nan)
            logp = np.where(np.log(p) != 0, np.log(p), np.nan)
            x = x_origin - 10.0 * np.log10(-logp)
            #####################################################################
            # Matplotlib pyplot figure and axes
            figure = plt.figure(figsize=plot_options["figsize"],layout="constrained")

            gs = plt.GridSpec(3, 3, figure=figure)
            psd_ax = figure.add_subplot(gs[0, :])
            apd_ax = figure.add_subplot(gs[1, 0])
            pvt_ax = figure.add_subplot(gs[1, 1:])
            pfp_ax = figure.add_subplot(gs[-1, :])

            #####################################################################
            # plot psd (sea ingest style)

            # Get noise floor values
            ktb = (
                (self.data_product.data["channel_metadata"].cal_temperature_degC + 273.15)
                .astype(np.float64)
                .mul(1.380649e-23)
                .apply(np.log10)
                .mul(10.0)
                .add(30.0)
                .add(self.data_product.data["channel_metadata"].cal_noise_figure_dB)
            ).droplevel("datetime")

            noise_floor_minfreqs = []
            noise_floor_maxfreqs = []
            noise_floor_dBmpHz = []

            # Construct for plotting
            for cf, nf in ktb.items():
                noise_floor_minfreqs.append((cf - 5e6) / 1e6)
                noise_floor_maxfreqs.append((cf + 5e6) / 1e6)
                noise_floor_dBmpHz.append(nf)

            # join all channel PSD results for plotting
            # and construct combined frequency axis
            full_band_freqs = []
            full_band_mean_psd = []
            full_band_max_psd = []
            for i, ch_psd in self.data_product.data["psd"].droplevel("datetime").iterrows():
                if i[1] == "max":
                    full_band_freqs.extend((ch_psd.index + i[0]).tolist())
                    full_band_max_psd.extend(ch_psd.tolist())
                elif i[1] == "mean":
                    full_band_mean_psd.extend(ch_psd.tolist())
            full_band_freqs = np.array(full_band_freqs) / 1e6  # Frequencies in MHz

            # Plot PSD
            psd_ax.plot(full_band_freqs, full_band_mean_psd, label="Mean",color='b')
            psd_ax.plot(full_band_freqs, full_band_max_psd, label="Max",color='r')
            psd_ax.set_xlim(full_band_freqs[0] - 2.5, full_band_freqs[-1] + 2.5)
            # Plot noise floors
            psd_ax.hlines(
                noise_floor_dBmpHz,
                noise_floor_minfreqs,
                noise_floor_maxfreqs,
                colors="k",
                linestyles="dashed",
                label="Noise Floor",
            )
            psd_ax.axvspan((channel_frequency / 1e6) - 5, (channel_frequency / 1e6) + 5, color="y", alpha=0.25, lw=0, label="Selected Channel")
            # Shade overloaded channels red
            first_ol = True
            for cf, ol in self.data_product.data["channel_metadata"].overload.droplevel("datetime").items():
                if ol:
                    if first_ol:
                        psd_ax.axvspan((cf / 1e6) - 5, (cf / 1e6) + 5, color="r", alpha=0.1, lw=0, label="Overload")
                        first_ol = False  # only label the shaded regions once
                    else:
                        psd_ax.axvspan((cf / 1e6) - 5, (cf / 1e6) + 5, color="r", alpha=0.1, lw=0)
            psd_ax.minorticks_on()
            psd_ax.grid(True, which="both", alpha=0.25)
            if plot_options["axes_labels"]:
                psd_ax.set_ylabel("PSD (dBm/Hz)")
                psd_ax.set_xlabel("Frequency (MHz)")
            sweep_time = (
                datetime.datetime.fromisoformat(self.data_product.data["sweep_metadata"].datetime[0][:-1])
                .strftime("%m-%d-%y %H:%M UTC")
            )
            if plot_options["plot_titles"]:
                psd_ax.set_title(f"Power Spectral Density")
            ######################################################################################
            # plot apd
            apd_ax.plot(x, apd.values)
            apd_ax.set_xticks(xtick, xtick_labels)
            apd_ax.set_xlim(xtick.min(), xtick.max())
            if plot_options["axes_labels"]:
                apd_ax.set_xlabel("Percent Exceeding Ordinate")
                apd_ax.set_ylabel("Power (dBm / 10 MHz)")
            if plot_options["plot_titles"]:
                apd_ax.set_title("APD")
            ###################################################################################
            # plot PVT
            x_pvt,pvt_max = self.data_product.get_xy_data(data_product='pvt',channel = channel,detector="maximum")
            x_pvt,pvt_rms = self.data_product.get_xy_data(data_product='pvt',channel = channel,detector="rms")
            pvt_ax.plot(x_pvt,pvt_max,color='r',alpha=1,linestyle="solid",label="Maximum")
            pvt_ax.plot(x_pvt,pvt_rms,color='b',alpha=1,linestyle="solid",label="Mean")
            if plot_options["axes_labels"]:
                pvt_ax.set_xlabel("Time (s)")
                pvt_ax.set_ylabel("Power (dBm/10 MHz)") 
            if plot_options["plot_titles"]:
                pvt_ax.set_title("Power Versus Time")
            ###################################################################################
            # plot pfp
            x_pfp,peak_max = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="max",detector="peak")
            x_pfp,peak_mean = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="mean",detector="peak")              
            x_pfp,peak_min = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="min",detector="peak")              
            x_pfp,rms_max = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="max",detector="rms")
            x_pfp,rms_mean = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="mean",detector="rms")              
            x_pfp,rms_min = self.data_product.get_xy_data(data_product='pfp',channel = channel,capture_statistic="min",detector="rms")   
            pfp_ax.plot(x_pfp,peak_mean,color='r',alpha=1,linestyle="dashed",label="Peak")
            pfp_ax.fill_between(x_pfp,peak_min,peak_max,color='r',alpha=.25)   
            pfp_ax.plot(x_pfp,rms_mean,color='b',alpha=1,linestyle="dashed",label="RMS")
            pfp_ax.fill_between(x_pfp,rms_min,rms_max,color='b',alpha=.25)  
            if plot_options["axes_labels"]:
                pfp_ax.set_xlabel("Time (s)")
                pfp_ax.set_ylabel("Power (dBm/10 MHz)")
            if plot_options["plot_titles"]:    
                pfp_ax.set_title("Periodic Frame Power")
            plt.tight_layout()                     
            if plot_options["grid"]:
                apd_ax.grid(True, "minor", "y", alpha=0.75)
                apd_ax.grid(True, "major", "both", alpha=0.5)
                pvt_ax.grid()
                pfp_ax.grid()
            if plot_options['legend']:
                psd_ax.legend()
            if plot_options["save"]:
                plt.savefig(plot_options["save"])
            if plot_options["show"]:
                plt.show()  
                
    def plot_gains(self,**options):
            """Plots the apd for the selected capture, provide the capture_id as a dictionary"""
            defaults = {"legend":False,
                        "save":False,
                        "show":True,
                        "grid":True}
            plot_options = {}
            for key,value in defaults.items():
                plot_options[key] = value
            for key,value in options.items():
                plot_options[key] = value
            gains  = self.data_product.gains
            plt.plot(gains.index/1e6,gains.values,"k.")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Gain (dB)")

            if plot_options["grid"]:
                plt.grid()
            if plot_options['legend']:
                plt.legend()
            if plot_options["save"]:
                plt.savefig(plot_options["save"])
            if plot_options["show"]:
                plt.show()      

    def plot_noise_figures(self,**options):
            """Plots the apd for the selected capture, provide the capture_id as a dictionary"""
            defaults = {"legend":False,
                        "save":False,
                        "show":True,
                        "grid":True}
            plot_options = {}
            for key,value in defaults.items():
                plot_options[key] = value
            for key,value in options.items():
                plot_options[key] = value
            noise_figures  = self.data_product.noise_figures
            plt.plot(noise_figures.index/1e6,noise_figures.values,"k.")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Noise Figure (dB)")
            if plot_options["grid"]:
                plt.grid()
            if plot_options['legend']:
                plt.legend()
            if plot_options["save"]:
                plt.savefig(plot_options["save"])
            if plot_options["show"]:
                plt.show()    

class DayBlockPlotter():
    def __init__(self,data) -> None:
        self.data_product  =data
    def plot_day(self,**options):
        for table_name in ("psd", "pfp", "pvt"):
            self.data_product.data[table_name] = self.data_product.data[table_name].astype("float32")
        detector_plot(self.data_product.data, self.data_product.sensor, watermark=True,**options) 

    def plot_day_psd(self,capture_statistic = "mean",**options):
        defaults = {"cmap":"viridis",
                    "cbar":True,
                    "save":False,
                    "show":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value
        assert capture_statistic in self.data_product.psd_statistics, " Capture statistic should be one of the options in self.data_product.psd_statistics "   
        psd_df =self.data_product.get_day_psd(capture_statistic)    
        plt.pcolormesh(1e-6*psd_df.columns.astype(float),psd_df.index,psd_df,cmap=plot_options["cmap"])
        if plot_options["cbar"]:
            pass
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Time (UTC)")
        plt.title(f"{capture_statistic} PSD for {self.data_product.sensor} on {self.data_product.date}")
        plt.tight_layout()
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show() 

    def plot_aligned_pfp(self,channel,capture_statistic='mean',detector='rms',**options):
        defaults = {"cmap":"viridis",
                    "cbar":True,
                    "save":False,
                    "show":True}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value
        if channel in self.data_product.frequencies:
            frequency = channel
        elif isinstance(channel,int):
            frequency = self.data_product.frequencies[channel]
        else:
            frequency = float(channel)
        aligned_pfp = self.data_product.get_aligned_pfp(channel=frequency,capture_statistic=capture_statistic,detector=detector)
        day_pfp_plot(aligned_pfp,frequency,self.data_product.sensor)
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()                 

class SummaryPlotter():
    def __init__(self,data) -> None:
        self.summary = data

    def plot_all_sensor_heatmaps(self,**options):
        defaults = {"legend":False,
                    "save":False,
                    "show":True,
                    "grid":True,
                    "sensors":self.summary.sensor_names,
                    "streams":['max','mean'],
                    "cmaps":['Reds','Blues','Greens'],
                    "time_min":None,
                    "time_max":None}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value                
        number_sensors = len(plot_options["sensors"])
        number_streams = len(plot_options["streams"])
        if number_sensors==1:
            figure, axes = plt.subplots(nrows=number_streams,ncols=number_sensors,sharey=True,sharex=True,figsize=(20,9))
            axes = np.reshape(axes,[number_streams,1])
        else:
            figure, axes = plt.subplots(nrows=number_streams,ncols=number_sensors,sharey=True,sharex=True,figsize=(20,9))
        for sensor_index,sensor in enumerate(plot_options["sensors"]):
            for stream_index,stream in enumerate(plot_options["streams"]):
                selected_data = self.summary.data[self.summary.data['sensor_name']==sensor]
                if plot_options["time_min"]:
                    time_min=plot_options["time_min"]
                    if isinstance(plot_options["time_min"],str):
                            time_min = pd.to_datetime(plot_options["time_min"],utc=True)
                            selected_data = selected_data[selected_data["acquisition_timestamp"]>time_min]
                    else:
                        selected_data = selected_data[selected_data["acquisition_timestamp"]>time_min]
                
                if plot_options["time_max"]:
                    time_max = plot_options["time_max"]
                    if isinstance(plot_options["time_max"],str):
                            time_min = pd.to_datetime(plot_options["time_max"],utc=True)
                            selected_data = selected_data[selected_data["acquisition_timestamp"]<time_max]
                    else:
                        selected_data = selected_data[selected_data["acquisition_timestamp"]<time_max]                       
                trimmed_data = selected_data[["acquisition_timestamp",'channel_frequency_mhz',stream]]
                start_time = min(trimmed_data["acquisition_timestamp"])
                stop_time = max(trimmed_data["acquisition_timestamp"])
                pivot = trimmed_data.pivot(index="acquisition_timestamp", columns="channel_frequency_mhz", values=stream)
                sns.heatmap(pivot,ax = axes[stream_index,sensor_index],cmap=plot_options["cmaps"][stream_index])
                axes[stream_index,sensor_index].invert_yaxis()
                locs = axes[stream_index,sensor_index].get_yticks()

                axes[stream_index,sensor_index].set_yticks([locs[0],locs[-1]],labels=(start_time.date(),stop_time.date()),rotation = 0)
                if stream_index==0:
                    axes[stream_index,sensor_index].set_title(sensor)
                axes[stream_index,sensor_index].set_xlabel(None)
                if stream_index==len(plot_options["streams"])-1:
                    axes[stream_index,sensor_index].set_xlabel("Frequency (MHz)")
                if sensor_index==0:
                    axes[stream_index,sensor_index].set_ylabel("Date")
                else:
                    axes[stream_index,sensor_index].set_ylabel(None)
                
        plt.tight_layout()
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()   
        return figure
    

    
    def plot_scatter_summary(self,**options):
        defaults = {"save":False,
                    "show":True,
                    "grid":True,
                    "alpha":.25}
        plot_options = {}
        for key,value in defaults.items():
            plot_options[key] = value
        for key,value in options.items():
            plot_options[key] = value           
        sns.jointplot(x = "mean", y = "max", 
              hue="overload",data = self.summary.data,alpha = plot_options["alpha"]) 
        if plot_options["grid"]:
            plt.grid()
        if plot_options["save"]:
            plt.savefig(plot_options["save"])
        if plot_options["show"]:
            plt.show()   
        figure = plt.gcf()
        return figure
# -----------------------------------------------------------------------------
# Module Scripts


def test_single_plotter(test_data = r".\test data\2024_6_14_21_37_1.sigmf",**options):
    data_product = DataProduct(test_data)
    plotter = DataProductPlotter(data_product)
    plotter.plot_all_psds(show=True)

def test_pfp_histogram(test_data = r".\test data\2024_6_14_21_37_1.sigmf",**options):
    data_product = DataProduct(test_data)
    plotter = DataProductPlotter(data_product)
    plotter.plot_channel_pfp_hist(channel=5,show=True)

def test_block_plotter(test_data = r".\test data\2024-7-4_seadog07.its.ntia.gov.zip",**options):
    data_product = DayBlock(file_path=test_data)
    plotter = DayBlockPlotter(data_product)
    plotter.plot_day(show=True)

def test_slice(test_data = r".\test data\2024-7-4_seadog07.its.ntia.gov.zip",slice_time ="2024-07-04 12:00" ,**options):
    data_product = DayBlock(file_path=test_data)
    plotter = DayBlockPlotter(data_product)
    plotter.plot_day(show=True,annotation_date = slice_time)
    slice = DataProduct(data_product.get_single(slice_time))
    single_plotter = DataProductPlotter(slice)
    single_plotter.plot_channel_summary(channel = slice.frequencies[10],show=True)

def test_summary_plotter(test_data = r".\test data\2024_7.csv"):
    summary = SummaryTable(test_data)
    plotter = SummaryPlotter(summary)
    plotter.plot_all_sensor_heatmaps(show =True,sensors = ["GMM"],streams=["max","mean","PAPR"])

def test_summary_plotter_2(test_data = r"D:\SEA\Tier 2\summaries\2024_9.csv"):
    summary = SummaryTable(test_data)
    plotter = SummaryPlotter(summary)
    plotter.plot_all_sensor_heatmaps(show =True,streams=["max","mean","PAPR"])

def test_summary_scatterplot(test_data =r".\test data\2024_7.csv"):
    summary = SummaryTable(test_data)
    plotter = SummaryPlotter(summary)
    plotter.plot_scatter_summary(show =True)

def test_psd_day_plot(test_data = r".\test data\2024-7-4_seadog07.its.ntia.gov.zip",capture_statistic="mean"):
    data_product = DayBlock(file_path=test_data)
    plotter = DayBlockPlotter(data_product)
    plotter.plot_day_psd(show=True,capture_statistic=capture_statistic)

def test_pfp_day_plot(test_data = r".\test data\2024-7-4_seadog07.its.ntia.gov.zip",channel =5,capture_statistic="mean"):
    data_product = DayBlock(file_path=test_data)
    plotter = DayBlockPlotter(data_product)
    plotter.plot_aligned_pfp(channel=channel,show=True,capture_statistic=capture_statistic)

def make_all_day_plots(top_directory,output_directory,exclude="seadog08",include=None,force_new=False):
    file_names = os.listdir(top_directory)
    if include:
        file_names = [file_name for file_name in file_names if re.search(include,file_name,re.IGNORECASE)]
    if exclude:
        file_names = [file_name for file_name in file_names if not re.search(exclude,file_name,re.IGNORECASE)]
    block_files = list(filter(lambda x:re.search(BLOCK_MONTH_PATTERN,x),file_names))
    destination_names = list(map(lambda x: os.path.join(output_directory,x),os.listdir(output_directory)))
    for file_name in block_files:
        match = re.search(DATABLOCK_PATTERN,file_name)
        sensor_name = match.groupdict()["sensor_hostname"]
        date = match.groupdict()["date"]
        if sensor_name in SENSOR_COMMON_NAMES.keys():
            sensor = SENSOR_COMMON_NAMES[sensor_name]
        elif sensor_name in SENSOR_COMMON_NAMES.values():
            sensor = sensor_name
        plot_name = os.path.join(output_directory, f"Day_Activity_inspect_{sensor}_doi_{date}.png")
        #print(f"sensor name is {sensor_name}, sensor date is {date}, \nplot name is {plot_name} \n"+"*"*80)
        if not force_new:
            if plot_name in destination_names:
                continue
        try:
            day_block = DayBlock(file_path=os.path.join(top_directory,file_name))
            plotter = DayBlockPlotter(day_block)
            plotter.plot_day(savename_pass = output_directory)
        except Exception as e:
            print(f"The plot has failed for {plot_name}")
            print(e)



def production_data_to_tier2(production_directory = r"C:\Users\sandersa\Box\Production Data",
                             tier2_directory=r"C:\Users\sandersa\Box\SEA-Tier2 - REV1",**options):
    """ This removes any pendleton data or oceana data, renames the files to date_common_name.zip and populates subdirectories of 
    PSD, PFP, and PVT"""
    defaults = {"exclude":"seadog08|seadog10",
                "include":None,
                "force_new":False,
                "psd_statistics":["mean","median","95th_percentile","99th_percentile","max"],
                "pfp_statistics":["mean"],
                "pfp_detectors":["rms"],
                "verbose":True}    
    tier2_options = {}
    for key,value in defaults.items():
        tier2_options[key] = value
    for key,value in options.items():
        tier2_options[key]=value    
    raw_directory = os.path.join(tier2_directory,"Raw Data")
    day_plot_directory = os.path.join(tier2_directory,"Day Plots")
    psd_directory = os.path.join(tier2_directory,"PSD")
    pfp_directory = os.path.join(tier2_directory,"PFP Aligned")
    summary_directory = os.path.join(tier2_directory,"Summaries")
    if tier2_options["verbose"]:
        print("Checking That Directories Exist")
    for directory in [raw_directory,day_plot_directory,psd_directory,pfp_directory,summary_directory]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            if tier2_options["verbose"]:
                print(f"Making {directory} ....")

    ###########################################################################################################
    # Move and rename the raw data files
    if tier2_options["verbose"]:
        print(f"Copy and Renaming Directories {datetime.datetime.now()}")    
    copy_and_rename(production_directory=production_directory,
                    output_directory=raw_directory,
                    exclude=tier2_options["exclude"],
                    include=tier2_options["include"],
                    force_new=tier2_options["force_new"])
    ###########################################################################################################
    # Make all Day plots
    if tier2_options["verbose"]:
        print(f"Making Day Plots {datetime.datetime.now()}")   
    make_all_day_plots(top_directory=raw_directory,
                       output_directory=day_plot_directory,
                       exclude=tier2_options["exclude"],
                       include=tier2_options["include"],
                       force_new=tier2_options["force_new"])
    ###########################################################################################################
    # Save all the psds
    if tier2_options["verbose"]:
        print(f"Saving all psds {datetime.datetime.now()}")   
    save_all_psd(top_directory=raw_directory,output_directory=psd_directory,
                capture_statistics=tier2_options["psd_statistics"],
                exclude=tier2_options["exclude"],
                include=tier2_options["include"],
                force_new=tier2_options["force_new"])
    ###########################################################################################################
    # Save all aligned pfps
    if tier2_options["verbose"]:
        print(f"Saving all aligned periodic frame powers {datetime.datetime.now()}")   
    save_all_pfp(top_directory=raw_directory,output_directory=pfp_directory,
                capture_statistics=tier2_options["pfp_statistics"],
                detectors=tier2_options["pfp_detectors"],
                exclude=tier2_options["exclude"],
                include=tier2_options["include"],
                force_new=tier2_options["force_new"])
    #############################################################################################################
    # Make monthly summaries
    if tier2_options["verbose"]:
        print(f"Summarizing by month {datetime.datetime.now()}")
    summarize_by_month(top_directory=raw_directory,output_directory=summary_directory,
                       include=tier2_options["include"],exclude=tier2_options["exclude"])
    
    if tier2_options["verbose"]:
        print(f"Finished building tier2 {datetime.datetime.now()}")
    
# -----------------------------------------------------------------------------
# Module Runner
if __name__=="__main__":
    #test_single_plotter()
    #test_pfp_histogram()
    #test_block_plotter()
    #test_slice()
    #test_summary_plotter_2()
    #make_all_day_plots("D:\SEA\Tier 2","D:\SEA\Tier 2\sensor_day_plots")
    #make_all_day_plots(r"D:\SEA\test_zip_files\test",r"D:\SEA\test_zip_files\test")
    #test_summary_scatterplot(r"D:\SEA\Tier 2\summaries\2024_9.csv")
    #test_psd_day_plot(capture_statistic="median")
    #test_pfp_day_plot(channel=9)
    # production_data_to_tier2(force_new = False)
    # production_data_to_tier2(tier2_directory=r"C:\Users\sandersa\Box\Oceana",include="Oceana|seadog10",exclude=None)
    # production_data_to_tier2(tier2_directory=r"C:\Users\sandersa\Box\Pendleton",include="Pendleton|seadog08",exclude=None)
    # summarize_by_month(top_directory=r"C:\Users\sandersa\Box\SEA-Tier2 - REV1\Raw Data",output_directory=r"C:\Users\sandersa\Box\SEA-Tier2 - REV1\Summaries",force_new=True)
    # #production_data_to_tier2(tier2_directory=r"\\kipp-smb.nist.gov\ctl\675\675-01\Internal NIST Collaboration\SEA Data\GMM",include="GMM|seadog07",exclude=None)
    # summarize_by_month(top_directory=r"C:\Users\sandersa\Box\Pendleton\Raw Data",output_directory=r"C:\Users\sandersa\Box\Pendleton\Summaries",
    #                   include="Pendleton|seadog08",exclude=None,force_new=True)
    # summarize_by_month(top_directory=r"C:\Users\sandersa\Box\Oceana\Raw Data",output_directory=r"C:\Users\sandersa\Box\Oceana\Summaries",
    #                   include="Oceana|seadog10",exclude=None,force_new=True)
    print("Currently this module does nothing when run as a script, but test functions are available to run individual pieces of functionality. See the function definitions for more details.")

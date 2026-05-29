# -----------------------------------------------------------------------------
# Name:      reports.py  
# Purpose:    To create shareable html reports for SEA data
# Authors:     aric.sanders@nist.gov
# Created:    2024-10-25 
# License:     NIST License
# -----------------------------------------------------------------------------
"""reports.py creates html reports for SEA sensor data
"""
# -----------------------------------------------------------------------------
# Standard Imports
import sys
import os
from collections import namedtuple
# -----------------------------------------------------------------------------
# Third Party Imports
sys.path.append(os.path.join(os.path.dirname( __file__ ),'.')) # if the repo is library/source/modules, one deep change to .
from data_models import *
from plotters import *
import folium
from folium import plugins
import panel as pn
pn.extension()
# -----------------------------------------------------------------------------
# Module Constants
SENSOR_LOCATIONS ={"Midway":[32.7139892, -117.175255],
                       "HU":[37.02578, -76.3412],
                       "NIT":[36.91552, -76.3225],
                       "GMM":[39.9918, -105.275],
                       "Pendleton":[33.28231, -117.38838],
                       "Catalina-Omni":[33.3795615, -118.4164147],
                       "Catalina-Directional":[33.3795615, -118.4164147],
                       "CBBT-Omni":[37.0460938, -76.0628427],
                       "CBBT-Directional":[37.0460938, -76.0628427],
                       "Oceana":[36.825105, -76.0295619]}

UNIQUE_SENSOR_LOCATIONS={"Midway":[32.7139892, -117.175255],
                       "HU":[37.02578, -76.3412],
                       "NIT":[36.91552, -76.3225],
                       "GMM":[39.9918, -105.275],
                       "Pendleton":[33.28231, -117.38838],
                       "Catalina":[33.3795615, -118.4164147],
                       "CBBT":[37.0460938, -76.0628427],
                       "Oceana":[36.825105, -76.0295619]}


SENSOR_HEATMAP_PATHS = {"Midway":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_9_Midway_50.0.npy'),
                       "HU":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_2_HU_65.5.npy'),
                       "NIT":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_1_NIT_48.5.npy'),
                       "GMM":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_11_GMM_6.0.npy'),
                       "Pendleton":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_8_USMC_Camp_Pendleton_6.4.npy'),
                       "Catalina-Omni":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_6_Mt_Orizaba_omni_4.5.npy'),
                       "Catalina-Directional":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_7_Mt_Orizaba_diretional_4.5.npy'),
                       "CBBT-Omni":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_3_CBBT_omni_40.0.npy'),
                       "CBBT-Directional":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_4_CBBT_directional_40.0.npy'),
                       "Oceana":os.path.join(os.path.dirname( __file__ ),'resources','20250107_aws_5_Oceana_40.0.npy')}

SensorInfo = namedtuple('SensorInfo', ['index', 'location', 'latitude', 'longitude', 'height',
    'antenna_filename', 'azimuth', 'downtilt', 'noise_floor','category_from_file','ofab_filename'])
# -----------------------------------------------------------------------------
# Module Functions
def load_heatmap_df(file_path):
    """Loads a npy file with lat, long and loss"""
    data = np.load(file_path, allow_pickle=True)
    sensor_info = data[0]
    #print(sensor_info)
    lat = data[2][0]
    lon = data[3][0]
    loss = data[4][0]
    output = np.array([lat,lon,loss]).T
    df = pd.DataFrame(output,columns=['lat','lon','loss'])
    return df

def downsample_map_points(data,digit=1.4):
    new_coordinates =[]
    new_xylosses = []
    for datum in data:
        new_point = [np.true_divide(np.rint(datum[0] * 10**digit), 10**digit),np.true_divide(np.rint(datum[1] * 10**digit), 10**digit)]
        new_xyloss = [np.true_divide(np.rint(datum[0] * 10**digit), 10**digit),np.true_divide(np.rint(datum[1] * 10**digit), 10**digit),datum[2]]
        if new_point in new_coordinates:
            continue
        else:
            new_coordinates.append(new_point)
            new_xylosses.append(new_xyloss)
    return new_xylosses

def load_downsampled_points(file_path,downsample_digit = 1.4):
    """Downsampled versison of load heatmap"""
    data = np.load(file_path, allow_pickle=True)
    sensor_info = data[0]
    #print(sensor_info)
    lat = data[2][0]
    lon = data[3][0]
    loss = data[4][0]    
    output = np.array([lat,lon,loss]).T
    output = downsample_map_points(output)
    df = pd.DataFrame(output,columns=['lat','lon','loss'])
    df['loss']=np.exp(-1*(df["loss"].abs()-df["loss"].abs().min())**2/(4*df["loss"].abs().max()))**2
    return df

def build_sensor_map(sensor_locations=SENSOR_LOCATIONS,**options):
    """Builds the sensor health and description page"""
    defaults = {"start_location":[40,-102],
                "start_zoom":4,
                "default_radius":80000,
                "height":500,
                "heatmaps":False,
                "heatmap_paths":SENSOR_HEATMAP_PATHS,
                "downsample":True,
                "sensor_colors_gradients":{}}
    sensor_map_options = {}
    for key,value in defaults.items():
        sensor_map_options[key]=  value
    for key,value in options.items():
        sensor_map_options[key] = value
    sensor_map= folium.Map(location=sensor_map_options["start_location"], zoom_start=sensor_map_options["start_zoom"])
    markdown_text = ""
    for sensor_name,sensor_location in sensor_locations.items():
        markdown_text += f"""Sensor Name:<span style = "color:red">{sensor_name}</span></h1>,Sensor Host Name:<span style = "color:red">{get_sensor_ip_from_name(sensor_name)}</span></h1>
        <hr/></br>"""
        
        folium.Marker(
            location=sensor_location,
            tooltip=sensor_name,
            icon=folium.Icon()).add_to(sensor_map)
        if sensor_map_options["default_radius"]:
            folium.vector_layers.Circle(location=sensor_location,
                                        radius =sensor_map_options['default_radius'], 
                                    color='red',opacity=.5).add_to(sensor_map)
        if sensor_map_options["heatmaps"]:
                try:
                    heatmap_path=sensor_map_options["heatmap_paths"][sensor_name]
                except:
                    continue
                if sensor_name in sensor_locations.keys():
                    if sensor_map_options["downsample"]:
                        df = load_downsampled_points(heatmap_path,downsample_digit =1.4)
                    else:
                        df = load_heatmap_df(heatmap_path)
                    plugins.HeatMap(df,min_opacity= ".5",max_opacity = "1",min_val=0,gradient={".1": "blue", ".2": "cyan", ".5": "lime", ".8": "yellow", "1": "red"}).add_to(sensor_map) #

    map_pane=pn.pane.plot.Folium(sensor_map, height=sensor_map_options["height"],name="Sensor Locations")

    return pn.Accordion(pn.pane.Markdown(markdown_text,name = "Sensor Names"),map_pane)

def build_sensor_name_map(sensor_locations=UNIQUE_SENSOR_LOCATIONS,**options):
    """Builds the sensor health and description page"""
    import folium
    from folium.features import DivIcon
    defaults = {"start_location":[40,-102],
                "start_zoom":4,
                "default_radius":80000,
                "height":500,
                "width":500}
    sensor_map_options = {}
    for key,value in defaults.items():
        sensor_map_options[key]=  value
    for key,value in options.items():
        sensor_map_options[key] = value
    sensor_map= folium.Map(location=sensor_map_options["start_location"], zoom_start=sensor_map_options["start_zoom"])

    for sensor_name,sensor_location in sensor_locations.items():
    ## This one looks good but doesn't touch the map in the right place
    #     folium.Marker(
    #         location=sensor_location,
    #         icon=folium.DivIcon(
    #                   html=f"""<div><p style=width: 50px;height:25px;
    # background-color: lightgray;
    # padding: 20px;
    # text-align: center;
    # margin: 50px;
    # box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);border: dashed black 2px;font-family: Arial; color:'black';"><h1>{sensor_name}</h1></p><i class="glyphicon glyphicon-star"></i></div>""")).add_to(sensor_map)

        folium.Marker(
            location=sensor_location,
            icon=DivIcon(        icon_size=(150, 36),
        icon_anchor=(75, 18), # Adjust to center the text
        html=f'<div style="font-size: 16pt; color: blue;">{sensor_name}</div>',
            )).add_to(sensor_map)

    map_pane=pn.pane.plot.Folium(sensor_map, height=sensor_map_options["height"],width=sensor_map_options["width"],name="Sensor Locations")

    return map_pane
# -----------------------------------------------------------------------------
# Module Classes
# -----------------------------------------------------------------------------
# Module Scripts
# -----------------------------------------------------------------------------
# Module Runner
if __name__=="__main__":
    #map_ = build_sensor_map(sensor_locations={"Oceana":[36.825105, -76.0295619]},start_location=[36.825105, -76.0295619],start_zoom=8)
    #map_ =build_sensor_map(default_radius=80000,heatmaps=False)
    #map_ = build_sensor_map(sensor_locations={"GMM":[39.9918, -105.275]},start_location=[39.9918, -105.275],start_zoom=8,default_radius=False,heatmaps=True,downsample=True)
    #map_ = build_sensor_map(sensor_locations={"Pendleton":[33.28231, -117.38838]},start_location=[33.28231, -117.38838],start_zoom=8,heatmaps=True,downsample=True)
    #map_ = build_sensor_map(heatmaps = False)
    map_ = build_sensor_name_map()
    map_.show()
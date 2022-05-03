#import pandas as pd
import streamlit as st

import pandas as pd
import numpy as np
import json
import ast
import datetime
import pandasql as ps
import geopandas as gpd

from datetime import datetime

import matplotlib.pyplot as plt

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import TimeSliderChoropleth
from branca.element import Figure

from streamlit_folium import folium_static


# Define a function that is basically a colormap between min and max value
# For displaying sentiment in colors
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 0#255 - b - r
    return r, g, b

# GOURCE only takes hex-values as colours
def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def add_hyphen(rgb):
    return '#' + rgb


epoch = datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() #* 1000.0


path = gpd.datasets.get_path('nybb')
df_polygon = gpd.read_file(path)
df_polygon = df_polygon.to_crs(epsg=4326)

# Project to NAD83 projected crs
df_polygon = df_polygon.to_crs(epsg=2263)

# Access the centroid attribute of each polygon
df_polygon['centroid'] = df_polygon.centroid

# Project to WGS84 geographic crs

# geometry (active) column
df_polygon = df_polygon.to_crs(epsg=4326)

# Centroid column
df_polygon['centroid'] = df_polygon['centroid'].to_crs(epsg=4326)


## Load data here (df_traffic)

df = pd.read_csv('data/traffic_data.csv')
# Align formatting of date columns and take non-null values
df = df.loc[~(df['borough_roadway'].isna())]
df.Date = pd.to_datetime(df.Date)
df['year'] = df.Date.dt.year
df['month'] = df.Date.dt.month
df['weekday'] = df.Date.dt.weekday
df['day_of_month'] = df.Date.dt.day

df_temp = df.copy() #[(df['year'] == 2020)]# & (df_res['borough_roadway'] == 'Brooklyn')]
df_temp = df_temp.groupby(['Date', 'borough_roadway'])[['Traffic_Volume']].sum()
df_temp = df_temp.reset_index(drop=False)

# We need to create bins
bins = np.linspace(min(df_temp['Traffic_Volume']), max(df_temp['Traffic_Volume']), 100)

color_labels = [add_hyphen(rgb_to_hex(rgb(bins[0], bins[-1], bins[x]))) for x in range(len(bins) - 1)]

df_temp['color'] = pd.cut(
    df_temp['Traffic_Volume'],
    bins,
    labels=color_labels,
    include_lowest=False
)

df_temp = df_temp.dropna()
df_temp1 = df_temp[['Date', 'borough_roadway', 'color', 'Traffic_Volume']]


df_temp1['unixtime'] = df_temp1.Date.apply(lambda x: unix_time_millis(x))

# Mapping to the BoroCode from df_polygon
borough_mapping = {
    'Bronx' : 2, 
    'Brooklyn' : 3, 
    'Manhattan' : 1, 
    'Queens' : 4, 
    'Staten Island' : 5,
}


df_temp1['Borough_ID'] = df_temp1.borough_roadway.map(borough_mapping)
df_temp1['Borough_ID'] = df_temp1['Borough_ID'].astype(str) # fixing formatting
df_temp1['unixtime'] = df_temp1['unixtime'].astype(int) # fixing formatting

# Prepare dictionary for TimeSliderChoropleth
# We want the following setup
# {BoroCode1 : {unixtime : {color : color_value, opacity : opacity}},
#  BoroCode2 : {unixtime : {color : color_value, opacity : opacity}},
#  etc...}

borough_dict={}
for i in df_temp1['Borough_ID'].unique():
    borough_dict[i]={}
    for j in df_temp1[df_temp1['Borough_ID'] == i].set_index(['Borough_ID']).values:   
        borough_dict[i][j[4]]={'color':j[2],'opacity':0.3} # 


#nyc_map = folium.Map(location=[40.70, -73.94], tiles='cartodbpositron', zoom_start=10)

fig = Figure(height=850,width=1000)
nyc_map = folium.Map([40.70, -73.94], tiles='cartodbpositron', zoom_start=10)
fig.add_child(nyc_map)

nyc_geojson = df_polygon.copy()
nyc_geojson = nyc_geojson[['geometry', 'BoroCode']]
nyc_geojson['BoroCode'] = nyc_geojson['BoroCode'].astype(str)

# Adding the borocode here and the styledict is the borough_dict
g = TimeSliderChoropleth(
    data=nyc_geojson.set_index('BoroCode').to_json(),
    styledict=borough_dict,
    overlay=True
).add_to(nyc_map)

for _, r in df_polygon.iterrows():
    lat = r['centroid'].y
    lon = r['centroid'].x
    folium.Marker(location=[lat, lon],
                  popup='Borough: {} <br> length: {} <br> area: {}'.format(r['BoroName'], 
                                                                           r['Shape_Leng'], 
                                                                           r['Shape_Area']
                                                                          ),
                  tooltip='<strong>See info about {}</strong>'.format(r['BoroName'])
                 ).add_to(nyc_map)


folium.LayerControl().add_to(nyc_map)

nyc_map

def app():
    st.markdown('Underneath here, you can see the NYC map of the traffic')

    folium_static(nyc_map)

import streamlit as st

# import libraries
import pandas as pd
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import show, output_notebook, reset_output
from bokeh.models import  ColumnDataSource, Legend
from bokeh.models import HoverTool
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,
    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5

#%matplotlib inline

## IMPORTANT 
# pip install streamlit-folium
import folium
from streamlit_folium import folium_static


# import warnings
# warnings.filterwarnings("ignore")

# df_sample = pd.read_csv('data/Drinking_Water_Quality_Distribution_Monitoring_Data.csv')
# df_sites = pd.read_excel('data/OpenData_Distribution_Water_Quality_Sampling_Sites_Updated_2021-0618.xlsx')
# df = pd.merge(df_sample, df_sites, on='Sample Site')


# source_crs = 'epsg:2263' # Coordinate system of the file from https://aem.run/posts/2021-07-02-learning-the-basics-of-gis-mapping-with-leaflet/
# target_crs = 'epsg:4326' # Global lat-lon coordinate system

# polar_to_latlon = pyproj.Transformer.from_crs(source_crs, target_crs)
# df['lat'], df['lon'] = polar_to_latlon.transform(df['X - Coordinate'], df['Y - Coordinate'])

# df.drop(columns=['X - Coordinate', 'Y - Coordinate'], inplace=True)

# df['Fluoride (mg/L)'].replace({'<0.3':'0.3'}, inplace=True)
# df['Fluoride (mg/L)'] = df['Fluoride (mg/L)'].astype('float')

# df['Turbidity (NTU)'].replace({'<0.10':'0.1'}, inplace=True)
# df['Turbidity (NTU)'] = df['Turbidity (NTU)'].astype('float')

# df['Coliform_float'] = df['Coliform (Quanti-Tray) (MPN /100mL)'].replace({'<1':'0', '>200.5':'201'})
# df['Coliform_float'] = df['Coliform_float'].astype('float')

# df['Ecoli_float'] = df['E.coli(Quanti-Tray) (MPN/100mL)'].replace({'<1':'0', '>200.5':'201'})
# df['Ecoli_float'] = df['Ecoli_float'].astype('float')

# df['Year'] = df['Sample Date'].str.split('/').str[2]
# df['Month'] = df['Sample Date'].str.split('/').str[0]
# df['Day'] = df['Sample Date'].str.split('/').str[1]

df = pd.read_csv('data/water_data_preprocessed.csv')

# apparently, there was a type-error
df['Sample Site'] = df['Sample Site'].astype(str)

### Borough allocation

for index, row in df.iterrows():
    if row['Sample Site'].startswith('2'):
        df.at[index, 'color'] = 'yellow'
        df.at[index, 'borough'] = 'Brooklyn'
    elif row['Sample Site'].startswith('5'):
        df.at[index, 'color'] = 'purple'
        df.at[index, 'borough'] = 'Staten Island'
    elif row['Sample Site'].startswith('4') or row['Sample Site'].startswith('7'):
        df.at[index, 'color'] = 'orange'
        df.at[index, 'borough'] = 'Queens'
    elif row['Sample Site'].startswith('1'):
        df.at[index, 'color'] = 'red'
        df.at[index, 'borough'] = 'Bronx'
    elif row['Sample Site'].startswith('3'):
        df.at[index, 'color'] = 'green'
        df.at[index, 'borough'] = 'Manhattan'
    else: 
        df.at[index, 'color'] = 'blue'

df_loc = df[['Sample Site', 'lat', 'lon', 'color', 'borough']]
df_loc = df_loc.drop_duplicates().reset_index()

ny_lat = 40.730610
ny_lon = -73.935242
ny_map = folium.Map(location=[ny_lat, ny_lon],
                    zoom_start = 10)

for index,row in df_loc.iterrows():
    folium.CircleMarker([row['lat'], row['lon']], popup=row['Sample Site'], color=row['color'],
            fill=True, opacity=0.5, radius = 2).add_to(ny_map)


def app():
    st.markdown('### **What do we all need for living? - Air, Water and Love right?**')
    st.markdown('Being able to drink water safely is a basic human right and the Sustainability Development Goal 6: Clean Water and Sanitation')
    st.markdown('This is why one of the main goals of our analysis is to investigate the water quality in NYC by analyzing data from the Water sampling stations in New York.')
      
    st.markdown('### **But how can you image these Water sampling stations in NYC and how do they work?**')
 
    st.video('https://www.youtube.com/watch?v=6YIZCVkfY5M')#
      
    st.markdown('###If you have been or live in NYC, you have probably seen them before! And now you know what they are for!')
    st.markdown('They are spread all over the city to check the water quality in every part of the city and every of the five boroughs')
    st.markdown('Below you can explore the exact locations of the sample stations and maybe find to the nearest to where you are living or staying to check it out the next time you walk by!´)

    folium_static(ny_map)

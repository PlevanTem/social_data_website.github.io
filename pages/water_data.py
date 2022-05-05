
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
import plotly.express as px

#%matplotlib inline

## IMPORTANT 
# pip install streamlit-folium
import folium
from streamlit_folium import folium_static


# import warnings
# warnings.filterwarnings("ignore")

df = pd.read_csv('data/Water_quality.csv')

df_loc = df[['Sample Site', 'lat', 'lon', 'color', 'borough']]
df_loc = df_loc.drop_duplicates().reset_index()

ny_lat = 40.730610
ny_lon = -73.935242
ny_map = folium.Map(location=[ny_lat, ny_lon],
                    zoom_start = 10)

for index,row in df_loc.iterrows():
    pop = row['Sample Site']
    bor = row['borough']
    folium.CircleMarker([row['lat'], row['lon']], popup=f'Sample Site: {pop}, Borough: {bor}', color=row['color'],
            fill=True, opacity=0.5, radius = 2).add_to(ny_map)

#fig = px.scatter_mapbox(df, lat="lat" , lon="lon", hover_name="Sample Site", color="Water_quality", animation_frame='Year - Month', mapbox_style='carto-positron', category_orders={'Year - Month':list(np.sort(df_water['Year - Month'].unique()))}, zoom=8)

fig = px.scatter(df, x='lat', y='lon', hover_name='Sample Site')

def app():
    st.markdown('### **What do we all need for living? - Air, Water and Love right?**')
    st.markdown('Being able to drink water safely is a basic human right and the Sustainability Development Goal 6: Clean Water and Sanitation')
    st.markdown('This is why one of the main goals of our analysis is to investigate the water quality in NYC by analyzing data from the Water sampling stations in New York.')
      
    st.markdown('### **But how can you image these Water sampling stations in NYC and how do they work?**')
 
    st.video('https://www.youtube.com/watch?v=6YIZCVkfY5M')
      
    st.markdown('### **If you have been or live in NYC, you have probably seen them before! And now you know what they are for!**')
    st.markdown('They are spread all over the city to check the water quality in every part of the city and every of the five boroughs')
    st.markdown('Below you can explore the exact locations of the sample stations and maybe find to the nearest to where you are living or staying to check it out the next time you walk by!')

    folium_static(ny_map)
    
    st.markdown('### **Now you have seen how the samples are taken and where the stations are - but what exactly is measured?**')
    st.markdown('There are the following 5 main indicators that are measured and important for the water quality.')
    st.markdown('* Residual Free Chlorine (mg/L)')
    st.markdown('* Turbidity (NTU)')
    st.markdown('* Fluoride (mg/L)')
    st.markdown('* Coliform (Quanti-Tray) (MPN /100mL)')
    st.markdown('* E.coli(Quanti-Tray) (MPN/100mL) ')

    st.header("Development of the water quality for the Sample Stations from 2015 - 2022")
    st.plotly_chart(fig)
    #st.plotly_chart(fig)

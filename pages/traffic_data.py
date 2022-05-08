import streamlit as st
from streamlit_folium import folium_static


import pandas as pd
import numpy as np
import json
import ast
import datetime
from tqdm.notebook import tqdm
import pandasql as ps
import geopandas as gpd

from datetime import datetime

import matplotlib.pyplot as plt


import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import TimeSliderChoropleth
from folium.plugins import HeatMapWithTime
from branca.element import Figure


import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from bokeh.plotting import figure
from bokeh.palettes import Spectral11
from bokeh.io import show, output_notebook, reset_output, curdoc
from bokeh.models import  ColumnDataSource, Legend, HoverTool
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.models.mappers import ColorMapper, LinearColorMapper

from branca.colormap import linear
from folium.plugins import TimeSliderChoropleth
import folium


########################
## DEFINING FUNCTIONS ##
########################

def plot_circle_multi(df,lw=2,pw=800,ph=500,t_str="hover,save,pan,box_zoom,reset",t_loc='above'):
    ''' Function to plot multiple lines with bokeh
    '''
    df.index = df.DATE
    df = df._get_numeric_data()
    source = ColumnDataSource(df)
    col_names = source.column_names[1:] #numeric column names

    p = figure(x_axis_type="datetime", 
            plot_width=pw, 
            plot_height=ph,
            y_range=(0,7000),
            y_axis_label='Traffic Volume',
            toolbar_location=t_loc, 
            tools=t_str, title="Traffic volume by district",)
    
    p_dict = dict() # store each line
    # set line color for each type
    numlines = len(col_names)
    mypalette=Spectral11[0:numlines]

    for col, c, col_name in zip(df.columns,mypalette,col_names):
        p_dict[col_name] = p.circle(x='DATE',y=col,source=source,color=c,line_width=lw)
        p.add_tools(HoverTool(
            renderers=[p_dict[col_name]],
            tooltips=[('Date','@DATE{%F}'),('Type', f'{col_name}'),('Traffic volume',f'@{col}')],
            formatters={'@DATE': 'datetime'}
        ))
    legend = Legend(items=[(x, [p_dict[x]]) for x in p_dict])
    p.add_layout(legend,'right')
    p.legend.click_policy = "mute"
    return p


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



########################
##   PREPROCESSING    ##
########################



df = pd.read_csv('data/traffic_data.csv')
df = df.rename(columns={'borough_roadway':'BOROUGH', 'Date':'DATE'})

# Align formatting of date columns and take non-null values
df = df.loc[~(df['BOROUGH'].isna())]
df.DATE = pd.to_datetime(df.DATE)
df['year'] = df.DATE.dt.year
df['month'] = df.DATE.dt.month
df['weekday'] = df.DATE.dt.weekday
df['day_of_month'] = df.DATE.dt.day



## For visualizing the traffic per borough
df_pr_borough = df.groupby('BOROUGH')['Traffic_Volume'].sum()
fig0 = px.bar(
    df_pr_borough,
    labels={
        'BOROUGH':'Borough',
        'value':'Traffic Volume',
    },
)
fig0.update_layout(
    title_text=f'''NYC total traffic measured from {df.DATE.min().strftime('%Y-%m-%d')} to {df.DATE.max().strftime('%Y-%m-%d')}''',
    legend_title="",)
fig0.show()



long_df = df.groupby(['Hour', 'BOROUGH'])['Traffic_Volume'].sum().reset_index()
fig1 = px.bar(
    long_df, 
    x="Hour", 
    y="Traffic_Volume", 
    color="BOROUGH",
    labels={
        'BOROUGH':'Borough',
        'Traffic_Volume':'Traffic Volume',
    },
)
fig1.update_layout(
    title_text=f'''NYC total traffic measured from {df.DATE.min().strftime('%Y-%m-%d')} to {df.DATE.max().strftime('%Y-%m-%d')} - Grouped by Borough''',
    legend_title="Borough",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 1
    )
)

fig1.show()


long_df = df.groupby(['year', 'BOROUGH'])['Traffic_Volume'].sum().reset_index()
fig2 = px.bar(
    long_df, 
    x="year", 
    y="Traffic_Volume", 
    color="BOROUGH",
    labels={
        'BOROUGH':'Borough',
        'Traffic_Volume':'Traffic Volume',
        'year':'Year'
    },
)
fig2.update_layout(
    title_text=f'''NYC total traffic per year - Grouped by Borough''',
    legend_title="Borough",
)

fig2.show()


df_day = df.groupby(['BOROUGH','DATE'])['Traffic_Volume'].sum().reset_index()
p1 = plot_circle_multi(df_day[df_day.BOROUGH == 'Bronx'].round())
p2 = plot_circle_multi(df_day[df_day.BOROUGH == 'Brooklyn'].round())
p3 = plot_circle_multi(df_day[df_day.BOROUGH == 'Manhattan'].round())
p4 = plot_circle_multi(df_day[df_day.BOROUGH == 'Queens'].round())
p5 = plot_circle_multi(df_day[df_day.BOROUGH == 'Staten Island'].round())

# set the layout for the plots and add them to three tabs for the three categories
tab1 = Panel(child=p1,title="Bronx")
tab2 = Panel(child=p2,title="Brooklyn")
tab3 = Panel(child=p3,title="Manhattan")
tab4 = Panel(child=p4,title="Queens")
tab5 = Panel(child=p5,title="Staten Island")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])
curdoc().add_root(tabs)

#show(tabs)


########################
##  VISUALISING MAP   ##
########################

df_traffic = df.copy()
df_traffic['Year - Month'] = df_traffic['year'].astype('str') + "-" + df_traffic['month'].astype('str')

import numpy as np
lat_long_list_2 = []
df_traffic = df_traffic.copy()
times = list(np.sort(df_traffic['Year - Month'].unique()))
for time in times:
    temp=[]
    for index, row in df_traffic[df_traffic['Year - Month'] == time].iterrows():
        temp.append([row['roadway_latitude'],
                     row['roadway_longitude'], 
                     row['Traffic_Volume']]
                   )
    lat_long_list_2.append(temp)


# ADDING POLYGONS
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

fig = folium.Figure(width=850,height=550)

ny_map = folium.Map(
    location=[40.70, -73.94],
    zoom_start=10,
    tiles='CartoDB positron'
)

fig.add_child(ny_map)

# Add the polygons representative of the boroughs
for _, r in df_polygon.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'orange'})
    folium.Popup(r['BoroName']).add_to(geo_j)
    geo_j.add_to(ny_map)

    
# Add centroids of boroughs to nyc_map
for _, r in df_polygon.iterrows():
    lat = r['centroid'].y
    lon = r['centroid'].x
    folium.Marker(location=[lat, lon],
                  popup='Borough: {} <br> length: {} <br> area: {}'.format(r['BoroName'], 
                                                                           r['Shape_Leng'], 
                                                                           r['Shape_Area']
                                                                          ),
                  tooltip='<strong>See info about {}</strong>'.format(r['BoroName'])
                 ).add_to(ny_map)
    
gradient = {.33: 'green', .66: 'brown', 1: 'red'}

HeatMapWithTime(
    lat_long_list_2,
    gradient=gradient,
    radius=5,
    auto_play=True,
    position='bottomright'
).add_to(ny_map)



def app():
    st.markdown('## Traffic Volume of all 5 NYC boroughs')
    st.plotly_chart(fig0)

    st.markdown('## Traffic Volume between the 5 NYC boroughs (during the day)')
    st.plotly_chart(fig1)

    st.markdown('## Traffic Volume between the 5 NYC boroughs (per year)')
    st.plotly_chart(fig2)

    st.markdown('## Traffic Volume between the 5 NYC boroughs (per date)')
    st.bokeh_chart(tabs, use_container_width=True)

    st.markdown('## Heatmap of Traffic Volume between the 5 NYC boroughs (per date)')
    folium_static(ny_map)
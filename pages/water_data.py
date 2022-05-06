
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
import folium
from folium.plugins import HeatMapWithTime
from branca.element import Figure

#%matplotlib inline

## IMPORTANT 
# pip install streamlit-folium
import folium
from streamlit_folium import folium_static


# import warnings
# warnings.filterwarnings("ignore")

df = pd.read_csv('data/Water_quality.csv')

# get an overview of the Sample Sites in the 5 boroughs in NCY

df_loc = df[['Sample Site', 'lat', 'lon', 'color', 'borough']]
df_loc = df_loc.drop_duplicates().reset_index()

# Folium map of the Sample Site locations
ny_map = folium.Map(location=[40.730610, -73.935242], zoom_start = 10)
for index,row in df_loc.iterrows():
    pop = row['Sample Site']
    bor = row['borough']
    folium.CircleMarker([row['lat'], row['lon']], popup=f'Sample Site: {pop}, Borough: {bor}', color=row['color'],
            fill=True, opacity=0.5, radius = 2).add_to(ny_map)

# Plot the development of the water quality over time for each sample site
df['Year - Month'] = df['Year'].astype('str') + "-" + df['Month'].astype('str')
fig_time = px.scatter_mapbox(df, lat="lat" , lon="lon", hover_name="Sample Site", color="Water_quality", opacity=0.5, animation_frame='Year - Month', 
                             mapbox_style='carto-positron', color_continuous_scale = ["red", "green"], category_orders={'Year - Month':list(np.sort(df['Year - Month'].unique()))}, zoom=8)
fig_time.show()


### Folium Heatmap with Time for the bad water quality samples in NCY

lat_long_list = []
df_water_bad = df[df.Water_quality == 0]
times = list(np.sort(df['Year - Month'].unique()))
for time in times:
    temp=[]
    for index,  row in df_water_bad[df_water_bad['Year - Month'] == time].iterrows():
        temp.append([row['lat'],row['lon']])
    lat_long_list.append(temp)
    
fig = folium.Figure(width=850,height=550)
ny_map_heat=folium.Map(location=[40.70, -73.94],zoom_start=10)
fig.add_child(ny_map_heat)
HeatMapWithTime(lat_long_list,radius=5,auto_play=True,position='bottomright').add_to(ny_map_heat)

### Bokeh plot of the different indicators per borough

def grouping(indicator):
    df_group = df.groupby('borough').agg({indicator:['sum','count']}).reset_index()
    df_group.columns = df_group.columns.droplevel()
    df_group.rename(columns={df_group.columns[0]: "borough" }, inplace = True)
    df_group['frac_good'] = df_group['sum']/df_group['count']
    df_group['frac_all'] = df_group['sum']/len(df[df[indicator]==1])
    cds = ColumnDataSource(df_group)
    return  df_group, cds
  
df_chlor, cds_chlor = grouping('Chlorine_level')
df_turb, cds_turb = grouping('Turbidity_level')
df_fluor, cds_fluor = grouping('Fluoride_level')
df_coli, cds_coli = grouping('Coliform_level')
df_ecoli, cds_ecoli = grouping('Ecoli_level')
df_water, cds_water = grouping('Water_quality')

list_of_bars = ['count', 'sum']
colors = ['red', 'green']
dict_legend = {'count':'No of all samples', 'sum':'No of good samples'}

# Create a general function for customizing legends
def customize_legend(p, items): 
    p.legend.visible = False
    legend = Legend(items=items, location="top_right")
    p.add_layout(legend, 'right')
    p.legend.click_policy = "mute"

# Create the general layout of the 3 plots with width, height, title and axes labels
p1 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for water quality", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_water.borough.values)
p2 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for turbidity level", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_turb.borough.values)
p3 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for chlorine level", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_chlor.borough.values)
p4 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for fluoride level", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_fluor.borough.values)
p5 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for coliform level", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_coli.borough.values)
p6 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for ecoli level", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_ecoli.borough.values)

# create empty dictionaries for the bars and lists for the legend items
bar1, bar2, bar3, bar4, bar5, bar6 = {}, {}, {}, {}, {}, {}
items1, items2, items3, items4, items5, items6 = [], [], [], [], [], []

# create the bars for the three plots
for indx,i in enumerate(list_of_bars):
    bar1[i] = p1.vbar(x='borough',  top=i, source=cds_water, legend_label=i, fill_color=colors[indx], width=0.5)
    items1.append((dict_legend[i], [bar1[i]]))
    bar2[i] = p2.vbar(x='borough',  top=i, source=cds_turb, legend_label=i, fill_color=colors[indx], width=0.5) 
    items2.append((dict_legend[i], [bar2[i]]))
    bar3[i] = p3.vbar(x='borough',  top=i, source=cds_chlor, legend_label=i, fill_color=colors[indx], width=0.5) 
    items3.append((dict_legend[i], [bar3[i]]))
    bar4[i] = p4.vbar(x='borough',  top=i, source=cds_fluor, legend_label=i, fill_color=colors[indx], width=0.5)
    items4.append((dict_legend[i], [bar4[i]]))
    bar5[i] = p5.vbar(x='borough',  top=i, source=cds_coli, legend_label=i, fill_color=colors[indx], width=0.5) 
    items5.append((dict_legend[i], [bar5[i]]))
    bar6[i] = p6.vbar(x='borough',  top=i, source=cds_ecoli, legend_label=i, fill_color=colors[indx], width=0.5) 
    items6.append((dict_legend[i], [bar6[i]]))
    
# customize the legend for the three plots
customize_legend(p1, items1)
customize_legend(p2, items2)
customize_legend(p3, items3)
customize_legend(p4, items4)
customize_legend(p5, items5)
customize_legend(p6, items6)
    
# set the layout for the plots and add them to three tabs for the three categories
l1 = layout([[p1]])
l2 = layout([[p2]])
l3 = layout([[p3]])
l4 = layout([[p4]])
l5 = layout([[p5]])
l6 = layout([[p6]])
tab1 = Panel(child=l1,title="Water quality")
tab2 = Panel(child=l2,title="Turbidity level")
tab3 = Panel(child=l3,title="Chlorine level")
tab4 = Panel(child=l4,title="Fluoride level")
tab5 = Panel(child=l5,title="Coliform level")
tab6 = Panel(child=l6,title="Ecoli level")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6])
curdoc().add_root(tabs)

# add hovertool for each plot with the fraction of recommitted crimes in the group and across all groups
p1.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar1['count']]))
p2.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar2['count']]))
p3.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar3['count']]))
p4.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar4['count']]))
p5.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar5['count']]))
p6.add_tools(HoverTool(tooltips=[('Fraction of good samples for district', '@frac_good{0.0000}'),
                    ('Fraction of good samples over all district', '@frac_all{0.0000}')], renderers=[bar6['count']]))


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
    st.markdown('There are the following 5 main indicators that are measured and their allowable level in drinking water according to the WHO')
    st.markdown('* Residual Free Chlorine (mg/L) - **5mg/L**')
    st.markdown('* Turbidity (NTU) - **1 NTU**')
    st.markdown('* Fluoride (mg/L) - **4 mg/L**')
    st.markdown('* Coliform (Quanti-Tray) (MPN/100mL) - **0 MPN/100ml**')
    st.markdown('* E.coli(Quanti-Tray) (MPN/100mL) - **0 MPN/100ml** ')
    
    st.markdown(
        """
        These allowable limits were used to devide the levels for each indicator classifying the water as either good below the allowed limits 
        from WHO (0) or bad being above the allowed limits (1).Also the overall indicator Water Quality was created that rates the overall water 
        quality as good (1) if none of the indicators is above the allowed limits and bad if even one of does. Below you can explore the 
        different indicators in the presented 5 districts of NYC.
        """
    )

    st.markdown('### **Number of good and bad quality samples based on different indicators for each borough**')
    st.bokeh_chart(tabs, use_container_width=True)

    st.header("Development of the water quality for the Sample Stations from 2015 - 2022")
    
    st.plotly_chart(fig_time)
    #folium_static(ny_map_heat)
    #st.plotly_chart(fig)

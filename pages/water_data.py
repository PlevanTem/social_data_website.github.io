
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

# Prepare year and month column
df['Year - Month'] = df['Year'].astype('str') + "-" + df['Month'].astype('str')
# sorted Year Month
sorted_year_month = ['2015-1', '2015-2', '2015-3', '2015-4', '2015-5', '2015-6', '2015-7', '2015-8', '2015-9', '2015-10', 
          '2015-11', '2015-12', '2016-1', '2016-2', '2016-3', '2016-4', '2016-5', '2016-6', '2016-7', '2016-8', 
          '2016-9', '2016-10', '2016-11', '2016-12', '2017-1', '2017-2', '2017-3', '2017-4', '2017-5', '2017-6', 
          '2017-7', '2017-8', '2017-9', '2017-10', '2017-11', '2017-12', '2018-1', '2018-2', '2018-3','2018-4', 
          '2018-5', '2018-6', '2018-7', '2018-8', '2018-9', '2018-10', '2018-11', '2018-12', '2019-1', '2019-2', 
          '2019-3', '2019-4', '2019-5', '2019-6', '2019-7', '2019-8', '2019-9', '2019-10', '2019-11', '2019-12',
          '2020-1', '2020-2', '2020-3', '2020-4', '2020-5', '2020-6', '2020-7', '2020-8', '2020-9', '2020-10', 
          '2020-11', '2020-12', '2021-1', '2021-2', '2021-3', '2021-4','2021-5', '2021-6', '2021-7', '2021-8', 
          '2021-9','2021-10', '2021-11','2021-12', '2022-1', '2022-2', '2022-3']


### Bokeh plot of the number of samples for different time periods

### prep the data 
df_no_samples_month_year  = df.groupby('Year - Month').agg({'Sample Number':'count'}).reset_index()
df_no_samples_day = df.groupby('Day').agg({'Sample Number':'count'}).reset_index()
df_no_samples_month = df.groupby('Month').agg({'Sample Number':'count'}).reset_index()
df_no_samples_year = df.groupby('Year').agg({'Sample Number':'count'}).reset_index()

cds_month_year = ColumnDataSource(df_no_samples_month_year)
cds_day = ColumnDataSource(df_no_samples_day)
cds_month = ColumnDataSource(df_no_samples_month)
cds_year = ColumnDataSource(df_no_samples_year)

list_of_bars = ['Sample Number']
colors = ['blue']
dict_legend = {'Sample Number':'No of samples'}

# Create a general function for customizing legends
def customize_legend(p, items): 
    p.legend.visible = False
    legend = Legend(items=items, location="bottom_center")
    p.add_layout(legend, 'below')
    p.legend.click_policy = "mute"

# Create the general layout of the 3 plots with width, height, title and axes labels
p11 = figure(plot_width = 900, plot_height = 500,title="Number of samples taken at Month and Year", 
            x_axis_label='Year - Month', y_axis_label='Number of samples', x_range=sorted_year_month)
p22 = figure(plot_width = 900, plot_height = 500,title="Number of samples taken at Day of the month", 
            x_axis_label='Day', y_axis_label='Number of samles')
p33 = figure(plot_width = 900, plot_height = 500,title="Number of samples taken at Month of the Year", 
            x_axis_label='Month', y_axis_label='Number of samples')
p44 = figure(plot_width = 900, plot_height = 500,title="Number of samples taken in each Year 2015-2022", 
            x_axis_label='Year', y_axis_label='Number of samples')


# create empty dictionaries for the bars and lists for the legend items
bar11, bar22, bar33, bar44 = {}, {}, {}, {}
items11, items22, items33, items44 = [], [],  [], []

# create the bars for the three plots
for indx,i in enumerate(list_of_bars):
    bar11[i] = p11.vbar(x='Year - Month',  top=i, source=cds_month_year, legend_label=i, fill_color=colors[indx], width=0.5)
    items1.append((dict_legend[i], [bar11[i]]))
    bar22[i] = p22.vbar(x='Day',  top=i, source=cds_day, legend_label=i, fill_color=colors[indx], width=0.5) 
    items2.append((dict_legend[i], [bar22[i]]))
    bar33[i] = p33.vbar(x='Month',  top=i, source=cds_month, legend_label=i, fill_color=colors[indx], width=0.5) 
    items33.append((dict_legend[i], [bar33[i]]))
    bar44[i] = p44.vbar(x='Year',  top=i, source=cds_year, legend_label=i, fill_color=colors[indx], width=0.5) 
    items44.append((dict_legend[i], [bar44[i]]))

p11.xaxis.major_label_orientation = "vertical"
# customize the legend for the three plots
customize_legend(p11, items11)
customize_legend(p22, items22)
customize_legend(p33, items33)
customize_legend(p44, items44)
    
# set the layout for the plots and add them to three tabs for the three categories
l11 = layout([[p11]])
l22 = layout([[p22]])
l33 = layout([[p33]])
l44 = layout([[p44]])

tab11 = Panel(child=l11,title="Samples per Month and Year")
tab22 = Panel(child=l22,title="Samples per Day of Month")
tab33 = Panel(child=l33,title="Samples per Month of Year")
tab44 = Panel(child=l44,title="Samples per Year")

tabs_all = Tabs(tabs=[tab11, tab22, tab33, tab44])
curdoc().add_root(tabs_all)

#display plot
show(tabs_all)

### Plot the development of the water quality over time for each sample site
fig_time = px.scatter_mapbox(df, lat="lat" , lon="lon", hover_name="Sample Site", color="Water_quality", opacity=0.5, animation_frame='Year - Month', 
                             mapbox_style='carto-positron', color_continuous_scale = ["red", "green"], category_orders={'Year - Month':sorted_year_month}, zoom=8)
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
    st.markdown(
      """
      Being able to drink water safely is a basic human right and the [Sustainability Development Goal 6: Clean Water and Sanitation](https://sdgs.un.org/goals/goal6).
      From 2000 until 2020, the number of humans with access to drinking water that is managed safely so safe and healthy to drink, increased by 2 billion, however there
      were still 2 billion people worldwide lacking safely managed drinking water services. From a developed countries such as the United States, you would expect that 
      the water quality is great and drinking water is safe. But is it really?
      
      Answering this question is one of the main goals of our analysis and therefore we will investigate the water quality from Water sampling stations in New York City!
      """
    )
      
    st.markdown('### **But how can you image these Water sampling stations in NYC and how do they work?**')
 
    st.video('https://www.youtube.com/watch?v=6YIZCVkfY5M')
      
    st.markdown(
      """
      **If you have been or live in NYC, you have probably seen them before! And now you know what they are for!**
      
      They are spread all over the city to check the water quality in every part of the city and every of the five boroughs
      Below you can explore the exact locations of the sample stations and maybe find to the nearest to where you are living or staying to check it out the next time you walk by!
      """
    )
    
    folium_static(ny_map)
    
    st.markdown('### **Now you have seen how the samples are taken and that the samples are collected manually - Are they really collected regularly?**')
    
    st.bokeh_chart(tabs_all, use_container_width=True)
    
    st.markdown('### **So we can see that sampels are regularly collected even though it is a manual process  - but what exactly is measured in these samples?**')
    st.markdown(
      """
      There are the following 5 main indicators that are measured and their allowable level in drinking water. You can follow the links to read more about the healthy limits in drinking water
      * Residual Free Chlorine (mg/L) - **[4mg/L](https://www.cdc.gov/healthywater/drinking/public/water_disinfection.html#:~:text=What%20are%20safe%20levels%20of,effects%20are%20unlikely%20to%2occur)**
      * Turbidity (NTU) - **[1 NTU](https://apps.who.int/iris/bitstream/handle/10665/254631/WHO-FWC-WSH-17.01-eng.pdf)**
      * Fluoride (mg/L) - **[4 mg/L]( https://www.epa.gov/sites/default/files/2015-10/documents/2011_fluoride_questionsanswers.pdf)**
      * Coliform (Quanti-Tray) (MPN/100mL) - **[0 MPN/100ml](https://www2.gnb.ca/content/dam/gnb/Departments/h-s/pdf/en/HealthyEnvironments/water/Coliforme.pdf)**
      * E.coli(Quanti-Tray) (MPN/100mL) - **[0 MPN/100ml](https://www2.gnb.ca/content/dam/gnb/Departments/h-s/pdf/en/HealthyEnvironments/water/Coliforme.pdf)**
      """
    )
    
    st.markdown(
        """
        These allowable limits were used to devide the levels for each indicator classifying the water as either good below the allowed limits (0) 
        or bad being above the allowed limits (1). Also the overall indicator Water Quality was created that rates the overall water 
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
    
    st.header("Fraction of bad water quality sample of the months of the year")
    #st.plotly_chart(plot_months)

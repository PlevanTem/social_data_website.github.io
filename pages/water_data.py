
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
import plotly.graph_objects as go

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
colors = ['#0000FF']
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
    items11.append((dict_legend[i], [bar11[i]]))
    bar22[i] = p22.vbar(x='Day',  top=i, source=cds_day, legend_label=i, fill_color=colors[indx], width=0.5) 
    items22.append((dict_legend[i], [bar22[i]]))
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

p22.add_tools(HoverTool(tooltips=[('Finding', 'Looks like the number of samples collected at each day of the month \
                                    is distributed equally, great! But there is a lot less samples on the  31st of the \
                                    month. Why? Just because only about every second month has a 31st so no wonder there \
                                    are less samples taken')]))

p33.add_tools(HoverTool(tooltips=[('Findings', 'Looks like the number of samples collected at each month of the year \
                                    is distributed equally, great! Only November has a smaller number of samples taken \
                                    but that is the influence of the missing November 2021')]))

p44.add_tools(HoverTool(tooltips=[('Findings', 'Looks like the number of samples collected at each year is around 15000 \
                                    for all years, great ! Only 2022 has a significant smaller amount of samples which is \
                                    because we are still in 2022')]))

#display plot
show(tabs_all)

### Plot the development of the water quality over time for each sample site
fig_time = px.scatter_mapbox(df, lat="lat" , lon="lon", hover_name="Sample Site", color="Water_quality", opacity=0.5, animation_frame='Year - Month', 
                             mapbox_style='carto-positron', color_continuous_scale = ['#FF0000', '#0000FF'], category_orders={'Year - Month':sorted_year_month}, zoom=8)
fig_time.show()

### Investigation of the peak frames of the plot above
def value_count(series):
    return series.value_counts()[0]

def time_group(time, time_name):
    df_time = df.groupby(time).agg({'Water_quality':['count', value_count]}).reset_index()
    df_time_bor = df.groupby([time, 'borough']).agg({'Water_quality':['count', value_count]}).reset_index()
    df_time.columns = df_time.columns.droplevel()
    df_time.columns = [time_name, 'count', 'bad_count']
    df_time_bor.columns = df_time_bor.columns.droplevel()
    df_time_bor.columns = [time_name, 'borough', 'count', 'bad_count']
    df_time['frac_bad'] = df_time['bad_count']/df_time['count']
    df_time['frac_all'] = df_time['bad_count']/len(df[df['Water_quality']==0])
    df_time_bor['frac_bad'] = df_time_bor['bad_count']/df_time_bor['count']
    df_time_bor['frac_all'] = df_time_bor['bad_count']/len(df[df['Water_quality']==0])
    return df_time, df_time_bor
  
df_month, df_month_bor= time_group('Month', 'Month')

plot_month = go.Figure(data=[go.Bar(
    name='All Boroughs',
    x=df_month['Month'],
    y=df_month['frac_bad']
),
    go.Bar(
    name='Bronx',
    x=df_month_bor[df_month_bor.borough == 'Bronx']['Month'],
    y=df_month_bor[df_month_bor.borough == 'Bronx']['frac_bad']
),  
    go.Bar(
    name='Brooklyn',
    x=df_month_bor[df_month_bor.borough == 'Brooklyn']['Month'],
    y=df_month_bor[df_month_bor.borough == 'Brooklyn']['frac_bad']
),  
    go.Bar(
    name='Staten Island',
    x=df_month_bor[df_month_bor.borough == 'Staten Island']['Month'],
    y=df_month_bor[df_month_bor.borough == 'Staten Island']['frac_bad']
),  
    go.Bar(
    name='Manhattan',
    x=df_month_bor[df_month_bor.borough == 'Manhattan']['Month'],
    y=df_month_bor[df_month_bor.borough == 'Manhattan']['frac_bad']
),  
    go.Bar(
    name='Queens',
    x=df_month_bor[df_month_bor.borough == 'Queens']['Month'],
    y=df_month_bor[df_month_bor.borough == 'Queens']['frac_bad']
),  
])

plot_month.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="All Boroughs",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True]},
                           {"title": "Fraction of water samples with insufficient water quality per month for all boroughs",
                            "label": ['bla', 'blub']
                            }]),
                dict(label="Bronx",
                     method="update",
                     args=[{"visible": [False, True, False, False, False, False]},
                           {"title": "Fraction of water samples with insufficient water quality per month for Bronx",
                            }]),
                dict(label='Brooklyn',
                     method="update",
                     args=[{"visible": [False, False, True, False, False, False]},
                           {"title": 'Fraction of water samples with insufficient water quality per month for Brooklyn',
                            }]),
                dict(label='Staten Island',
                     method="update",
                     args=[{"visible": [False, False, False, True, False, False]},
                           {"title": 'Fraction of water samples with insufficient water quality per month for Staten Island',
                            }]),
                dict(label="Manhattan",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, False]},
                           {"title": "Fraction of water samples with insufficient water quality per month for Manhattan",
                            }]),
                dict(label="Queens",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, True]},
                           {"title": "Fraction of water samples with insufficient water quality per month for Queens",
                            }]),
            ]),
        )
    ])

plot_month.update_layout(
    title = "Fraction of water samples with insufficient water quality per month for all boroughs",
    xaxis_title="Month",
    yaxis_title="Fraction of bad water samples",
    legend_title="Boroughs",
)
  
plot_month.show()
  
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
colors = ['#FF0000', '#0000FF']
dict_legend = {'count':'No of all samples', 'sum':'No of good samples'}

# Create a general function for customizing legends
def customize_legend(p, items): 
    p.legend.visible = False
    legend = Legend(items=items, location="top_right")
    p.add_layout(legend, 'right')
    p.legend.click_policy = "mute"

# Create the general layout of the 3 plots with width, height, title and axes labels
p2 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for water quality", 
            x_axis_label='District', y_axis_label='Number of samples in the data', x_range=df_water.borough.values)
p1 = figure(plot_width = 950, plot_height = 500,title="Number of samples by district for turbidity level", 
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
    bar2[i] = p2.vbar(x='borough',  top=i, source=cds_water, legend_label=i, fill_color=colors[indx], width=0.5)
    items2.append((dict_legend[i], [bar2[i]]))
    bar1[i] = p1.vbar(x='borough',  top=i, source=cds_turb, legend_label=i, fill_color=colors[indx], width=0.5) 
    items1.append((dict_legend[i], [bar1[i]]))
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
tab2 = Panel(child=l2,title="Water quality")
tab1 = Panel(child=l1,title="Turbidity level")
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

# Turbidity influence on water quality

def time_ind_group(time, time_name, indicator):
    df_time = df.groupby(time).agg({indicator:['count', value_count]}).reset_index()
    df_time.columns = df_time.columns.droplevel()
    df_time.columns = [time_name, 'count', 'bad_count']
    df_time['frac_bad'] = df_time['bad_count']/df_time['count']
    df_time['frac_all'] = df_time['bad_count']/len(df[df[indicator]==0])
    return df_time
  
df_wq = time_ind_group('Month', 'Month', 'Water_quality')
df_tb = time_ind_group('Month', 'Month', 'Turbidity_level')

fig_turb = plt.figure(figsize=(15,7))
plt.bar(df_wq['Month'], df_wq['frac_bad'], color='blue', label='Water quality')
plt.bar(df_tb['Month'], df_tb['frac_bad'], color='orange', label='Turbidity level')
plt.title('Influence of turbidity on the overall water quality', fontsize=14)
plt.xlabel('Month', fontsize=1)
plt.ylabel('Fraction of water samples with insufficient quality', fontsize=14)
plt.legend()

# Overall trend over the years
df_year, df_year_bor = time_group('Year', 'Year')
for index, row in df.iterrows():
    if row['Turbidity (NTU)']>5:
        df.at[index, 'Turbidity_level_new']=0
    else:
        df.at[index, 'Turbidity_level_new']=1
        
for index, row in df.iterrows():
    if row['Chlorine_level'] == 0 or row['Turbidity_level_new'] == 0 or row['Fluoride_level'] == 0 or \
    row['Coliform_level']== 0 or row['Ecoli_level'] == 0:
        df.at[index, 'Water_quality_new'] = 0
    else:
        df.at[index, 'Water_quality_new'] = 1
       
df_year_new = time_ind_group('Year', 'Year', 'Water_quality_new')

fig_years, ax_years = plt.subplots(1, 2, figsize=(15,7))
ax_years[0].bar(df_year['Year'], df_year['frac_bad'], color='#0000FF')
ax_years[0].set_title('Development of bad quality water samples with Turbidity limit of 1 NTU')
ax_years[0].set_xlabel('Year', fontsize=14)
ax_years[0].set_ylabel('Fraction of samples with insufficient water quality', fontsize=14)
ax_years[1].bar(df_year_new['Year'], df_year_new['frac_bad'], color='#0000FF')
ax_years[1].set_title('Development of bad quality water samples with Turbidity limit of 5 NTU')
ax_years[1].set_xlabel('Year', fontsize=14)
ax_years[1].set_ylabel('Fraction of samples with insufficient water quality', fontsize=14)
ax_years[1].set_ylim(0, 0.25)
ax_years[1].axhline(0.01, color='#FF0000')

def app():
    st.markdown('''<a id='water'></a>### **What do we all need for living? - Air, Water and Love right?**''', unsafe_allow_html=True)
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
    st.markdown(
      """
      Below you can explore the water sample collection for different time frames to see if the manual collection is taking place consistently throughout the 
      months of the timeframe, days of the month, months of the year and for the years 2015 - 2022. 
      
      Overall, you can see that the manual data collection is taking place regularly. So, it'S convincing to see that the water quality is monitored and controlled 
      consistently to ensure New York City's population access to clean drinking water and health. There is a slight trend of more water samples being collected 
      in the summer months and a little less samples in the winter months. As can be seen from the video the samples are collected from stations on the streets 
      outside so the weather could be an explanation for these fluctuations but they are only small and in most months between 1200 and 1400 samples are collected.
      
      However there is a big gap in November 2021. Even though the NYC Department employee in the video stated that they are out every day of the year collecting
      samples even after a hurricane, that might not be always be the case. There was a series of storms and hurricane warnings in 
      [NYC in November 2021](https://www.nytimes.com/2021/11/13/nyregion/tornado-storm-long-island-nyc.html) which could explain the lack of sample collection in that month.
      This example highlights the weaknesses of the manual water sampling collection and in the long-term the consistent water quality monitoring needs to be ensured. 
      """
    )

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
    
    st.header("Development of the water quality for the Sample Stations from 2015 - 2022")
    st.markdown(
        """
        Below you can see the development of the water quality for each sample station from 2015 - 2022 with insufficient water quality with one of the indicators 
        being above the allowable limits indicated in red and all indicators being within the presented limits in blue. You might ask yourself here, why not 
        red and green to show the "bad" and "good" samples? These colors might not be distinct enough for people with color blindness so we chose colors with the 
        help of a [Color blindness similator](https://www.color-blindness.com/coblis-color-blindness-simulator/) that are also colorblind friendly so that the 
        distinction can be made by everyone!
        
        Overall, it can be seen that there is a positive development of more and more water samples with indicators in the allowable limits over the years and
        most of the samples indicate a sufficient quality of drinking water in NYC.
        """
    )
    
    st.plotly_chart(fig_time)

    st.header("But what about the times with a significant amount of insufficient water qualities in the samples?")
    st.markdown(
        """
        If you follow the time development in the figure above closely, you can notice that there seem to be a lot of water samples with insufficient water quality
        at the beginning of the year. To get a clearer view on this, see the visualization below of the fraction of samples with bad water quality below. 
        The fraction of the samples was taken here to take into account that there was fewer samples taken in November for example as discover before.
        
        The plot clearly shows that the insufficient water quality rises from the beginning of the year until it peaks in March. After that the water quality
        drastically improves until the end of the year. This development can be observed in all five boroughs so it is a phenomena across New York City.
        But what influences the measure of the water quality the most and so what is the explanation behind this development?
        """
    )
    st.plotly_chart(plot_month)
    
    st.markdown('### **What indicator influences the water quality most and contributes to an overall insufficient water quality?**')
    st.markdown(
        """
        In the plot below you can investigate the fraction of samples in and above the allowable limits for every indicator and the overall water quality. 
        
        What you can easily see after taking a look at the indicators is that the turbidity levels that are above the allowable limits in orange influence 
        the water quality the most in each district. All of the other indicators are mostly within the allowable limits. 
        """
    )
    st.bokeh_chart(tabs, use_container_width=True)
    
    st.header('Influence of Turbidity on the Water quality')
    st.markdown(
        """
        First of all, what exactly is turbidity?
        >> Turbidity is a measure of cloudiness of the water. Turbidity is monitored because it is a good indicator of water quality, 
        because high turbidity can hinder the effectiveness of disinfection, and because it is a good indicator of the effectiveness 
        of our filtration system. - [NYC Environmental Protection](https://www1.nyc.gov/assets/dep/downloads/pdf/water/drinking-water/drinking-water-supply-quality-report/2019-drinking-water-supply-quality-report.pdf)
        
        Factors that influence turbidity are the following and you can read more about it [here](https://www.tnrd.ca/services/water-sewage/chlorination-and-turbidity/)
        * storms
        * high rainfall
        * snow melt
        
       So a clear explanation for the high turbidity levels in March is the spring runoff of the snow after winter. 
        """
    )
    
    st.pyplot(fig_turb)
    
    st.header('Overall development of Water quality in NYC from 2015 - 2022')
    st.markdown(
        """
        **So are the exceedings of turbidity levels concerning? No, for a turbidity level between 1 and 5 NTU, the threats to the public's health are modest.**
        
        Overall, there has been a positive development in water quality since 2015 as can be seen in the left figure below
        and with the continous monitoring and NYC commitment to the [SDGs](https://www1.nyc.gov/site/international/programs/global-vision-urban-action.page) 
        this trend should continue. Especially because the climate change leads to higher turbidity levels due to more and more intense rains, 
        stronger storms or higher river levels according to the 
        [United States Environmental Protection Agency](https://www.epa.gov/arc-x/climate-adaptation-and-erosion-sedimentation) and because turbidity can
        hinder the effectiveness of water disinfection, it is important to take measures if consistent higher levels of turbidity are observed.
        
        When taking into account the limit of 5 NTU for Turbidity where the risks are modest, you can observe from the right figure below that the fraction 
        of water samples with insufficient quality has been consistently below 1% in each year from 2015 - 2022.
        """
    )
    st.pyplot(fig_years)
    
    st.markdown('**If you want to know more about the water in NYC, check out the NY state water [website](https://water.ny.gov/doh2/applinks/waterqual/#/home)!**')

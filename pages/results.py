import pandas as pd
import streamlit as st
import geojson

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import folium
from streamlit_folium import folium_static


import datetime
from pandas.io.formats.format import Datetime64Formatter

# # Water quality data preprocessing
# df_water = pd.read_csv('data/Water_quality.csv')
# df_water['Year - Month'] = df_water['Year'].astype('str') + "-" + df_water['Month'].astype('str')
# df_water['Year - Month'] = pd.to_datetime(df_water['Year - Month']).dt.to_period('M')

# df_water_bad = df_water[df_water.Water_quality == 0]
# df_water_monthly = df_water_bad.groupby(['Year - Month','borough'])[['Residual Free Chlorine (mg/L)']].mean().reset_index()


# # Traffic data preprocessing
# df_traffic = pd.read_csv('data/traffic_data.csv', parse_dates=['Date'])
# df_traffic['Year - Month'] = df_traffic['Date'].dt.to_period('M')

# df_traffic_day = df_traffic.groupby(['Date', 'borough_roadway'])['Traffic_Volume'].sum().reset_index(drop=False)
# df_traffic_day['Year - Month'] = pd.to_datetime(df_traffic_day['Date'], format='%Y/%m/%d').dt.to_period('M')

# df_traffic_day_monthly = df_traffic_day.groupby(['Year - Month','borough_roadway']).mean().reset_index()
# df_traffic_day_monthly.rename(columns={'borough_roadway':'borough'},inplace='True')


# # Recycling data preprocessing
# df_recycling = pd.read_csv('data/Total_collection_of_NYC_from_2015_to_2021.csv',index_col=0)
# df_recycling['Year - Month'] = pd.to_datetime(df_recycling['DATE'], format='%Y/%m/%d').dt.to_period('M')

# df_recycling_by_month = df_recycling.groupby(['Year - Month','BOROUGH']).mean().reset_index()
# df_recycling_by_month.drop(columns = 'BOR_CD',inplace=True)
# df_recycling_by_month.rename(columns={'BOROUGH':'borough'},inplace='True')

# # Limiting date to same time range. Based on the 3 dataframes above, it should be from 2015-02 to 2021-05
# df_recycling_by_month = df_recycling_by_month[
#     (df_recycling_by_month['Year - Month'] >= '2015-02') & 
#     (df_recycling_by_month['Year - Month'] <= '2021-05')
# ]

# df_traffic_day_monthly = df_traffic_day_monthly[
#     (df_traffic_day_monthly['Year - Month'] >= '2015-02') & 
#     (df_traffic_day_monthly['Year - Month'] <= '2021-05')
# ]

# df_water_monthly = df_water_monthly[
#     (df_water_monthly['Year - Month'] >= '2015-02') & 
#     (df_water_monthly['Year - Month'] <= '2021-05')
# ]

# # Merging
# temp = pd.merge(df_recycling_by_month, 
#                 df_traffic_day_monthly, 
#                 how='outer', 
#                 on=['Year - Month', 'borough']
#                )

# df_merged = pd.merge(df_water_monthly, 
#                      temp, 
#                      how='left', 
#                      on=['Year - Month', 'borough']
#                     )

# df_merged['date_str'] = df_merged['Year - Month'].apply(lambda x: str(x))

# df_merged.dropna(inplace=True)

# Read df_merged
# df_merged = pd.read_csv('data/df_merged.csv')



# with open('data/Borough Boundaries.geojson') as f:
#     gj = geojson.load(f)


# text = df_merged.apply(
#     lambda row: f"<br>Total collected:{round(row['TOTALCOLLECTED'],2)}T<br>Traffic Volume:{round(row['Traffic_Volume'],2)}", axis=1),



# # set plotly default theme
# pio.templates.default = 'plotly_white'

# figsize=(20, 10)

# fig = px.choropleth_mapbox(df_merged,
#           geojson=gj, 
#           locations=df_merged.borough,
#           featureidkey="properties.boro_name",
#           center = {"lat": 40.70, "lon": -73.94},
#           mapbox_style="carto-positron",
#           opacity=0.5,
#           color='Residual Free Chlorine (mg/L)',
#           animation_group="borough",
#           hover_name="borough",
#           hover_data=text,
#           color_continuous_scale=px.colors.sequential.deep,
#           zoom=9,
#           range_color=(0,1.2),
#           title='Water Quality in New York City in combination with Recycling data and Traffic data',
#           animation_frame="date_str")

# fig.update_layout(
#     title_text='Water Quality in New York City in combination with Recycling data and Traffic data'
# )

# fig.show()


def app():
    st.markdown('Something here')

    st.markdown('## Heatmap of Water Quality, Recycling Data & Traffic Volume (oh my god')
    #st.plotly_chart(fig, use_container_width=True, height=600)
    
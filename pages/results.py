import pandas as pd
import streamlit as st
import geojson
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import folium
from streamlit_folium import folium_static


import datetime
from pandas.io.formats.format import Datetime64Formatter

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# Read df_merged
df_merged = pd.read_csv('data/df_merged.csv')

scaler = MinMaxScaler()

scalable_columns = [
    'Residual Free Chlorine (mg/L)',
    'Turbidity (NTU)',
    'Fluoride (mg/L)',
    'Coliform_float',
    'Ecoli_float',
    'TOTALCOLLECTED',
    'Traffic_Volume'
]


df_merged_scaled = df_merged.copy()
df_merged_scaled[scalable_columns] \
    = scaler.fit_transform(df_merged_scaled[scalable_columns])


cmap = sns.diverging_palette(
    250, 
    15, 
    s=75, 
    l=40, 
    n=9, 
    center='light', 
    as_cmap=True
)

matrix = df_merged_scaled.corr(method='pearson')

# Create a mask
mask = np.triu(np.ones_like(matrix, dtype=bool))

fig_heatmap, ax = plt.subplots(figsize=(7, 7))
fig_heatmap = sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax)

boroughs = sorted(df_merged.borough.unique().tolist())




with open('data/Borough Boundaries.geojson') as f:
    gj = geojson.load(f)


text = df_merged.apply(
    lambda row: f"<br>Total collected:{round(row['TOTALCOLLECTED'],2)}T<br>Traffic Volume:{round(row['Traffic_Volume'],2)}", axis=1),


# set plotly default theme
pio.templates.default = 'plotly_white'

figsize=(20, 10)

fig = px.choropleth_mapbox(df_merged,
          geojson=gj, 
          locations=df_merged.borough,
          featureidkey="properties.boro_name",
          center = {"lat": 40.70, "lon": -73.94},
          mapbox_style="carto-positron",
          opacity=0.5,
          color='Residual Free Chlorine (mg/L)',
          animation_group="borough",
          hover_name="borough",
          hover_data=text,
          color_continuous_scale=px.colors.sequential.deep,
          zoom=9,
          range_color=(0,1.2),
          title='Water Quality in New York City in combination with Recycling data and Traffic data',
          animation_frame="date_str")

fig.update_layout(
    title_text='Water Quality in New York City in combination with Recycling data and Traffic data'
)

# fig.show()







def app():
    st.markdown('Something here')

    st.pyplot(fig_heatmap)

    with st.expander("See the correlation plots for each borough"):
        for borough in boroughs:
            cmap = sns.diverging_palette(
                250, 
                15, 
                s=75, 
                l=40, 
                n=9, 
                center='light', 
                as_cmap=True
            )

            matrix = df_merged_scaled[df_merged_scaled.borough == borough].corr(method='pearson')

            # Create a mask
            mask = np.triu(np.ones_like(matrix, dtype=bool))

            fig = sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f")
            st.markdown(f'Showing correlation for {borough}')
            st.pyplot(fig)


    st.markdown('## Heatmap of Water Quality, Recycling Data & Traffic Volume (oh my god')
    
    st.plotly_chart(fig, use_container_width=True, height=600)
    
    
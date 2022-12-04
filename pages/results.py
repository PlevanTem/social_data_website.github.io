import pandas as pd
import streamlit as st
import geojson
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import matplotlib.pyplot as plt

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

fig_heatmap, ax = plt.subplots()
sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax)

boroughs = sorted(df_merged.borough.unique().tolist())




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
    st.markdown(
        """
        Something here
        """
    )


    st.markdown(
        """
        We want to create a cool machine learning model that will be albe to tell us something about whether traffic and recycling has an impact on the water quality in New York City
        
        Let us then have a look at how the different columns in our data correlate to one another, such that we can make a decision on whether a model for this data would be any good:
        """
    )

    st.markdown(
        """
        Correlation plot of all the columns in our `df_merged`
        """
    )
    st.write(fig_heatmap)

    st.markdown(
        """
        **Finding**:

        From the correlation plot above, we can conclude that we see no significant correlation between the features. However, let's try to break it down per borough and see whether we find anything else useful in our analysis:
        """
    )

    with st.expander("See the correlation plots for each borough"):
        # 1
        cmap = sns.diverging_palette(
            250, 
            15, 
            s=75, 
            l=40, 
            n=9, 
            center='light', 
            as_cmap=True
        )

        matrix = df_merged_scaled[df_merged_scaled.borough == boroughs[0]].corr(method='pearson')

        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        p1, ax1 = plt.subplots()
        sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax1)
        st.markdown(f'Showing correlation for {boroughs[0]}')
        st.write(p1)


        # 2
        cmap = sns.diverging_palette(
            250, 
            15, 
            s=75, 
            l=40, 
            n=9, 
            center='light', 
            as_cmap=True
        )

        matrix = df_merged_scaled[df_merged_scaled.borough == boroughs[1]].corr(method='pearson')

        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        p2, ax2 = plt.subplots()
        sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax2)
        st.markdown(f'Showing correlation for {boroughs[1]}')
        st.write(p2)


        # 3
        cmap = sns.diverging_palette(
            250, 
            15, 
            s=75, 
            l=40, 
            n=9, 
            center='light', 
            as_cmap=True
        )

        matrix = df_merged_scaled[df_merged_scaled.borough == boroughs[2]].corr(method='pearson')

        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        p3, ax3 = plt.subplots()
        sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax3)
        st.markdown(f'Showing correlation for {boroughs[2]}')
        st.write(p3)


        # 4
        cmap = sns.diverging_palette(
            250, 
            15, 
            s=75, 
            l=40, 
            n=9, 
            center='light', 
            as_cmap=True
        )

        matrix = df_merged_scaled[df_merged_scaled.borough == boroughs[3]].corr(method='pearson')

        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        p4, ax4 = plt.subplots()
        sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax4)
        st.markdown(f'Showing correlation for {boroughs[3]}')
        st.write(p4)


        # 5
        cmap = sns.diverging_palette(
            250, 
            15, 
            s=75, 
            l=40, 
            n=9, 
            center='light', 
            as_cmap=True
        )

        matrix = df_merged_scaled[df_merged_scaled.borough == boroughs[4]].corr(method='pearson')

        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        p5, ax5 = plt.subplots()
        sns.heatmap(matrix, mask=mask, cmap=cmap, square=True, annot=True, fmt=".2f", ax=ax5)
        st.markdown(f'Showing correlation for {boroughs[4]}')
        st.write(p5)

    st.markdown(
        """
        **Findings**:

        When we break down the correlation for each borough, we do see some interesting correlation between the recycling data, the traffic data and the water quality. 
        This is basically what we wanted to show and thus what we wanted conduct a thorough analysis of, however, because of the limitations due to lack of data, we cannot say for sure that there in fact is a correlation between the features shown.

        Due to limited resources (data-wise and time-wise), we were not able to find a dataset (or multiple datasets) that held enough data for us to conclude on an analysis like the above.
        """
    )

    # st.markdown(
    #     """
    #     ## Development of water quality, recycling and traffic from 2015 to 2022!
        
    #     Please stand by for the plot below to load (it takes about 10 seconds 😅)
    #     """
    # )

    # st.plotly_chart(fig, use_container_width=True, height=600)
    
    # However, research have shown that traffic and the amount of waste can have an impact of the water quality of certain neighbouthoods / boroughs.


    st.markdown(
        """
        **Water data**:
        - Even though it's a manual sampling process (collection), the water data is measured continuously - There is one gap in 2021, but other than that, throughout the months and years, the data is pretty consistently sampled. New York City is really engaged in ensuring a good water quality, so at least for SGD 8, NYC is committed to having clean drinking water. 

        - We discovered one interesting pattern in the data - the water quality is usually the worst in March due to elevated turbidity levels. Investigating the possible reasons for the rise in turbidity levels, we found that the rise was due to an increase in temperature, which melted the snow from the winter months, adding dirt and particles to the water. Especially in the future, it's really important to keep an eye on the turbitity levels, because climate change leading to an increase in temperature, storms and more rainfalls lead to a continous increase of the turbidity in the water, minimizing the effect of sanitation.

        - Overall, the drinking water quality in New York City is satisfying with only below 1\% of the samples being outside the range of safe drinking water.


        **Recycling data**:
        - There are regional disparities on trash and disposal collection and recycling amount between different boroughs and community districts in New York City. We found that trash or refuse accounted for above 70% of collection tons and recyclable materials like paper metal, glass, plastic, and beverage cartons still need a more effective way to collect and reuse. Moreover, from 2015 to 2022, the total collection amount moderately incresaed in all boroughs, which indicates people cosumed more and more. This is what we need to pay attention to, be careful of consumerism.


        - In addition, we can see the two most populated boroughs(Queen and Manhatten) have the most recycled bins, which is a good thing and seems make sense. But, the least population borough, Staten Island collected the most trash and disposals with the least number of recycled bins! Plus, from the map, we found that freedom of trash bin is not realised in a lot of districts. If NYC put an effort on recycled infrastrucre, especially trash bins, this will be a helpful to step further on zero waste and sustainable development!


        **Traffic data**:
        - Even though the traffic data revealed nothing out of the ordinary from what you would expect (we added exstraordinary plots instead), we do still see a massive use of cars and vehicles in New York City. With the point made from the conclusion from the water data, an increasing use of cars do lead to an increase of green house gas emissions, and in the long run, the increased traffic will also lead to higher levels or turbidity in the drinking water (directly in contradiction to the [13th Sustainable Development Goal](https://sdgs.un.org/goals/goal13)). 

        **Overall**:
        - There is an overall positive trend of the water quality being better over the years. We found that there is the most amount of trash where the least amount of people live, which is counter-intuitive, however, New York City has really put an effort into collecting trash and recycling in the areas with the most trash. With the things that we found, we tried to make the best out of it - meaning that we wanted to find a relation between the amount of traffic, the amount of trash collected and the water quality, but the results were not promising, leaving our analysis being inconclusive.
        """
    )
    
    st.markdown(
        """
        ## The End.

        Now that you have seen our story, you can also visit our [GitHub page](https://github.com/ChristianRoesselDTU/social_data_website.github.io) to see how we did it! There, amongst other things, you will find a link to our [Notebook behind the scenes](https://github.com/ChristianRoesselDTU/social_data_website.github.io/raw/main/Explainer_Notebook_final.ipynb.zip) and [the data necessary to run it](https://github.com/ChristianRoesselDTU/social_data_website.github.io/raw/main/data.zip) (The notebook needs to be in the same folder as the data folder to run).
        """

    )

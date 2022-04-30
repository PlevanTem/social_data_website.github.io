import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Custom imports 
from multipage import MultiPage
from pages import main_page, recycling_data, traffic_data, water_data

# Wide page
st.set_page_config(page_title="Social Data And Visualization", page_icon="🐍", layout="wide", initial_sidebar_state="auto", menu_items=None,) 

# Create an instance of the app 
app = MultiPage()

# Title of the main page
t1, t2 = st.columns((0.5,0.5))  
t1.image('images/new_york_traffic.jpg')
t2.title("Exploring Water Quality in New York City")

with st.expander("ℹ️ - About this app", expanded=False):
    st.write(
    """     
    You can write something here that expands, possibly about how to interpret your plots
    """
)

# Add all your application here
app.add_page("Main Page", main_page.app)
app.add_page("Recycling Data", recycling_data.app)
app.add_page("Traffic Data", traffic_data.app)
app.add_page("Water Data", water_data.app)


# The main app
app.run()
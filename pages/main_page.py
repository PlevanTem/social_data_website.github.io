import pandas as pd
import streamlit as st


def app():
    st.markdown('Something here')
    
    df = pd.read_csv('data/traffic_data_2015_2020_latlong_borough.csv')
    df = df.head(5)

    st.table(df)
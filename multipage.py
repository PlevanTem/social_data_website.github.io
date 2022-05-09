"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):


        # for four buttons (at the top)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            but1 = st.button('Water Data')
            if but1:
                page = self.pages[0]

        with col2:
            but2 = st.button('Recycling Data')
            if but2:
                page = self.pages[1]

        with col3:
            but3 = st.button('Traffic Data')
            if but3:
                page = self.pages[2]

        with col4:
            but4 = st.button('Results')
            if but4:
                page = self.pages[3]


        # Dropdown to select the page to run  
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )
        
        # run the app function 
        page['function']()
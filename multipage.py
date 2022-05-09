"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st
import time

my_bar = st.progress(0)

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
        # Dropdown to select the page to run  
        page = st.sidebar.radio(
            'Website Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        if page == self.pages[0]:
            my_bar.progress(25)

        if page == self.pages[1]:
            my_bar.progress(50)

        if page == self.pages[2]:
            my_bar.progress(75)

        if page == self.pages[3]:
            my_bar.progress(100)


        # run the app function 
        page['function']()
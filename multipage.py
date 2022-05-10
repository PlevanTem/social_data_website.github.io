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

        # st.markdown(
        #     """
        #     Website progress
        #     """
        # )

        #The idea behind this website setup is for the reader to go through the pages in the following order
        #The reason why we chose this format was to find a good balance between an author-driven and a reader-driven story so that we could convey our message clearly and understandably but also allow the reader to get into more details and find some additional facts and details to discover.

        with st.expander("How to read this website", expanded=True):
            st.write(
            """     
            ## How to read this website
            
            Are you curious about the sanity of the water you drink everyday? We took at look at whether water quality is being affected by the traffic and recycling initiatives in New York City.

            To get the best experience of this website, you should go through the pages in the followign order:
            1. `Water Data` 
            2. `Recycling Data` 
            3. `Traffic Data`
            4. `Results`


            We hope you enjoy the journey of our website 😊

            If you are cusious about what is happening behind the scenes, take a look at our [GitHub page](https://github.com/ChristianRoesselDTU/social_data_website.github.io) 🐍 - ssssss

            (Click to hide)
            """
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
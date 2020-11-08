import streamlit as st
import pandas as pd
import altair as alt

####################### helper functions #######################

@st.cache  # add caching so we load the data only once
def load_data():
    # placeholder
    penguins_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/v0.1.0/inst/extdata/penguins.csv"
    return pd.read_csv(penguins_url)

def draw_title():
    st.markdown(
    """
        # Donation statistics analysis

        Team members:
        - Xinwen Liu (xinwenl), Xinyu Lin (xinyulin), Yuxi Luo (yuxiluo), Shaobo Guan (shaobog)

        Github repository:
        - [vectorization](https://github.com/CMU-IDS-2020/fp-vectorization)
    """
    )

def draw_narrative():
    # call other functions for narrative here
    st.markdown(
    """
        ### Narrative
        [here goes description]
    """
    )
    return

def draw_model():
    # call other functions for model here
    st.markdown(
    """
        ### Model: Project description <-> fully funded duration
        [here goes description]
    """
    )
    return

####################### end of helper functions #######################

draw_title()
draw_narrative()
draw_model()

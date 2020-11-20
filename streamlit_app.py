import streamlit as st
import pandas as pd
import altair as alt
from vega_datasets import data

####################### helper functions #######################

# @st.cache  add caching so we load the data only once
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

        [Github repository](https://github.com/CMU-IDS-2020/fp-vectorization)

        [Design Review slides](https://github.com/CMU-IDS-2020/fp-vectorization/documents/DesignReview.pdf)
    """
    )

def draw_narrative():
    # call other functions for narrative here
    st.markdown(
    """
        ## Visualization

    """
    )
    st.write("Donation request count; Proposed donation sum; Proposed donation mean")
    draw_v1()
    st.write("Free lunch percentage of schools which requested donation at least once")
    draw_v2()
    st.write("Mean of donation records; Sum of donation records")
    draw_v3()
    draw_v4()
    return

def draw_model():
    # call other functions for model here
    st.markdown(
    """
        ## Model: Predicting project fully-funded status

        Please see [logit.ipynb](https://github.com/CMU-IDS-2020/fp-vectorization/blob/main/modeling/logit.ipynb) for current state of our model
    """
    )
    return

####################### end of helper functions #######################

####################### visualization sections  #######################
def draw_v1():
    # slider for selecting specific years
    cnt_df = pd.read_csv("data/loc_time_join.csv")

    sum_avg_df = pd.read_csv("data/avg_loc_time_join.csv")

    year = st.slider("Year",
                     min_value = 2013,
                     max_value = 2018,
                     step = 1)

    # A slider filter
    # year_slider = alt.binding_range(min=2013, max=2018, step=1)
    # slider_selection = alt.selection_single(bind=year_slider, fields=['Post year'])

    # get count agg by the given year range
    cnt_filter_df = cnt_df[cnt_df['Post year'] == year]

    sum_avg_filter_df = sum_avg_df[sum_avg_df['Post Year'] == year]
    #make a map
    states = alt.topo_feature(data.us_10m.url, 'states')

    count_v = alt.Chart(
        states
    ).mark_geoshape(
        stroke='#aaa',
        strokeWidth=0.25
    ).encode(
        color = 'Count:Q',
        tooltip=['State:N', 'Count:Q']
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(cnt_filter_df, 'id', ['Count', 'State'])
    ).project(
        type='albersUsa'
    ).properties(
        width=400,
        height=200
    )

    sum_v = alt.Chart(
        states
    ).mark_geoshape(
        stroke='#aaa',
        strokeWidth=0.25
    ).encode(
        color = 'Sum:Q',
        tooltip=['State:N', 'Sum:Q']
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(sum_avg_filter_df, 'id', ['Sum', 'State'])
    ).project(
        type='albersUsa'
    ).properties(
        width=400,
        height=200
    )

    avg_v = alt.Chart(
        states
    ).mark_geoshape(
        stroke='#aaa',
        strokeWidth=0.25
    ).encode(
        color = 'Average:Q',
        tooltip=['State:N', 'Average:Q']
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(sum_avg_filter_df, 'id', ['Average', 'State'])
    ).project(
        type='albersUsa'
    ).properties(
        width=400,
        height=200
    )
    st.write(count_v)
    st.write(sum_v)
    st.write(avg_v)
    return

def draw_v2():
    df = pd.read_csv("data/free_lunch_state_metro.csv")
    v2 = alt.Chart(df).mark_point().encode(
            x='School State:N',
            y='School Metro Type:N',
            size='School Percentage Free Lunch:Q',
            color='School Percentage Free Lunch:Q',
            tooltip=['School State',
                 'School Metro Type',
                 'School Percentage Free Lunch']
        ).properties(
            width=1200,
        )
    st.write(v2)
    return

def draw_v3():
    df = pd.read_csv("data/state_donate_avg_sum.csv")
    slider = alt.binding_range(min=2013, max=2018, step=1)
    select_year = alt.selection_single(name="Year", fields=['Year'],
                                       bind=slider, init={'Year': 2013})

    mean_ = alt.Chart(df).mark_bar().encode(
        x=alt.X('Donor State:N'),
        y=alt.Y('Mean:Q'),
        tooltip = ['Donor State:N', 'Mean:Q']
    ).add_selection(
        select_year
    ).transform_filter(
        select_year
    )
    sum_ = alt.Chart(df).mark_bar().encode(
        x=alt.X('Donor State:N'),
        y=alt.Y('Sum:Q'),
        tooltip = ['Donor State:N', 'Sum:Q'],
    ).add_selection(
        select_year
    ).transform_filter(
        select_year
    )

    st.write(mean_)
    st.write(sum_)
    return

def draw_v4():
    grade=pd.read_csv("data/grade_by_year.csv")
    resource=pd.read_csv("data/resource_by_year.csv")

    slider = alt.binding_range(min=2013, max=2018, step=1)
    select_year = alt.selection_single(name="Year", fields=['Post year'],
                                       bind=slider, init={'Post year': 2018})
    g_histo = alt.Chart(grade).mark_bar().encode(
        x= "Project Grade Level Category:N",
        y= "Project Cost:Q",
        color=alt.Color("Project Grade Level Category:N"),
        tooltip=["Project Grade Level Category:N", "Project Cost:Q"]
    ).add_selection(
        select_year
    ).transform_filter(
        select_year
    ).properties(
        width=500,
        height=500
    )

    r_histo = alt.Chart(resource).mark_bar().encode(
        x= "Project Resource Category:N",
        y= "Project Cost:Q",
        color=alt.Color("Project Resource Category:N"),
        tooltip=["Project Resource Category:N", "Project Cost:Q"]
    ).add_selection(
        select_year
    ).transform_filter(
        select_year
    ).properties(
        width=500,
        height=500
    )
    st.write(g_histo)
    st.write(r_histo)
    return
#################### end of visualization sections ####################


draw_title()
draw_narrative()
draw_model()

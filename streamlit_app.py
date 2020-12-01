import re
import datetime
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from joblib import dump, load
from vega_datasets import data
from sklearn.linear_model import LogisticRegression

####################### global variables #######################

subcategories = ['Literacy', 'Mathematics', 'Literature & Writing', 'Special Needs',
   'Applied Sciences', 'Health & Wellness', 'Visual Arts',
   'Environmental Science', 'Early Development', 'ESL',
   'Health & Life Science', 'Music', 'History & Geography',
   'Character Education', 'College & Career Prep', 'Other',
   'Gym & Fitness', 'Performing Arts', 'Team Sports', 'Social Sciences',
   'Care & Hunger', 'Warmth', 'Extracurricular', 'Foreign Languages',
   'Civics & Government', 'Parent Involvement', 'Financial Literacy',
   'Nutrition Education', 'Community Service', 'Economics']

resources = ['Supplies', 'Technology', 'Books', 'Computers & Tablets',
   'Educational Kits & Games', 'Instructional Technology',
   'Reading Nooks, Desks & Storage', 'Flexible Seating', 'Trips',
   'Classroom Basics', 'Other', 'Art Supplies', 'Lab Equipment',
   'Sports & Exercise Equipment', 'Food, Clothing & Hygiene',
   'Musical Instruments', 'Visitors']

feature_words = ['student', 'school', 'learn', 'classroom', 'help', 'work', 'read',
   'love', 'day', 'class', 'skill', 'book', 'technology', 'time', 'one',
   'math', 'material', 'grade', 'children', 'different', 'project',
   'teach', 'like', 'world', 'create', 'best', 'learners', 'science',
   'education', 'community', 'language', 'home', 'activities', 'free',
   'access', 'opportunity', 'life', 'first', 'fun', 'hard', 'environment',
   'lunch', 'resource', 'experience', 'opportunities', 'excited',
   'diverse', 'eager', 'play', 'art', 'challenge', 'creative', 'goal',
   'music', 'amazing', 'social', 'poverty', 'games', 'hands-on',
   'research', 'knowledge', 'engaging', 'safe', 'computer', 'literacy',
   'reduced', 'however', 'comfortable', 'band', 'instrument', 'musical',
   'healthy', 'breakfast', 'hungry', 'team', 'sport', 'hurricane',
   'health', 'volleyball', 'basketball', 'soccer', 'college', 'museum',
   'paint', 'activity']

english_numbers = ["One", "Two", "Three", "Four", "Five",
                   "Six", "Seven", "Eight", "Nine", "Ten",
                   "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
                   "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
                   "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety",
                   "Thousand", "Million", "Billion"]

sample = "Imagine having tables and having to stand to play and complete work activities. I  teach preschool special education in an urban school district.  Most of the students are on free and reduced lunch.  My students are between the ages of 3-5 years old.  My students have a variety of disabilities including Autism, Down Syndrome, and language impairments. With the chairs, my students will be able to sit down to play with table toys like Legos and stringing beads.  My students will also be able to sit to complete work activities such as cutting with scissors, tracing their names and completing letter and number crafts. Donations will help my children sit and concentrate on their work.  My students will be better prepared for kindergarten.  They will be able to sit and learn basic skills such as colors, shapes, numbers and letters."

regex_numbers = re.compile("(\d|(" + "|".join(map(re.escape, english_numbers)) + "))")
clf = load('modeling/logit.joblib')

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

        [Design Review slides](https://github.com/CMU-IDS-2020/fp-vectorization/blob/main/documents/DesignReview.pdf)
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
    """
    )

    model_proj_desc_interaction()

    return

####################### model sections  #######################

def model_proj_desc_interaction():
    st.markdown(
    """
        ### Predicting based on entered project description
    """
    )

    # https://docs.streamlit.io/en/latest/api.html#streamlit.beta_columns
    description = st.text_area("Project description", value=sample) #, value="Input your project description here!")
    col1, col2, col3 = st.beta_columns([1, 1, 1])
    start = col1.date_input("Project start date", datetime.date(2017, 1, 1))
    end = col2.date_input("Project end date", datetime.date(2017, 5, 1))
    cost = col3.number_input("Project cost")

    col1, col2 = st.beta_columns([1, 1])
    subcat = col1.multiselect("Project category", subcategories)
    rescat = col2.multiselect("Resource category", resources)

    submitted = st.button("Submit project proposal")
    if submitted:
        st.markdown(
        """
            > Running model prediction
        """
        )

        # process the input to form a single-row dataframe for prediction
        df = pd.read_pickle("modeling/features.pkl")
        df = df.append(pd.Series(), ignore_index = True)
        df['X_essay_len'] = len(description)
        df['X_essay_?!'] = description.count('\\?') + description.count('\\!')
        df['X_essay_numbers'] = len(re.findall(regex_numbers, description))
        for s in subcategories:
            df["SUBCAT " + s] = description.count(s)
        for r in resources:
            df["RESCAT " + r] = description.count(r)
        df['Project Cost'] = cost
        df['Project Valid Time'] = (end-start).days
        df['Project Year 2016'] = start.year == 2016
        df['Project Year 2017'] = start.year == 2017
        df['Teacher Project Posted Sequence'] = 1 # default?
        for w in feature_words:
            df[w] = description.count(w)
        # print(df)

        # compute the model prediction result here
        predicted = clf.predict(df)[0]
        if predicted:
            st.write("Model prediction: this project proposal can be fully funded")
        else:
            st.write("Model prediction: this project proposal can not be fully funded")


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


####################### main #######################

sections = {
    'Description': draw_title,
    'Narrative': draw_narrative,
    'Model': draw_model,
}
option = st.sidebar.selectbox(
    "Sections", list(sections.keys())
)
sections[option]()

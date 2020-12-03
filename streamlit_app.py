import re
import datetime
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from joblib import dump, load
from vega_datasets import data
from sklearn.linear_model import LogisticRegression
import streamlit.components.v1 as components
from htbuilder import HtmlElement, div, span, styles
from htbuilder.units import px, rem, em

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

lr_weights = {'X_essay_len': 0.0003453689601916409,
 'X_essay_?!': 0.021064285398078284,
 'X_essay_numbers': 0.016932228938762856,
 'Project Valid Time': 0.0036698372667219735,
 'Project Cost': -0.0008740402700159638,
 'Teacher Project Posted Sequence': 0.0034669254479514256,
 'Project Year 2016': -0.25166270855080325,
 'Project Year 2017': -0.05981350582210004,
 'SUBCAT Literacy': -0.02707468032324999,
 'SUBCAT Mathematics': -0.02687656482146483,
 'SUBCAT Literature & Writing': -0.038273617655681796,
 'SUBCAT Special Needs': -0.026392979534569778,
 'SUBCAT Applied Sciences': 0.02549676901064158,
 'SUBCAT Health & Wellness': 0.0032012685338489405,
 'SUBCAT Visual Arts': 0.006071517684001481,
 'SUBCAT Environmental Science': 0.01884535546982471,
 'SUBCAT Early Development': -0.0006788974197114126,
 'SUBCAT ESL': 0.0023238287072987127,
 'SUBCAT Health & Life Science': 0.0018845667885582487,
 'SUBCAT Music': 0.013348892470044968,
 'SUBCAT History & Geography': -0.0007833087166925236,
 'SUBCAT Character Education': 0.004330765985309406,
 'SUBCAT College & Career Prep': -0.0028378513009307206,
 'SUBCAT Other': -0.006017752496990515,
 'SUBCAT Gym & Fitness': 0.005922847859095761,
 'SUBCAT Performing Arts': 0.006202877206921002,
 'SUBCAT Team Sports': 0.013268170269707941,
 'SUBCAT Social Sciences': 0.00041355381631686034,
 'SUBCAT Care & Hunger': 0.023784772780853423,
 'SUBCAT Warmth': 0.023784772780853423,
 'SUBCAT Extracurricular': 0.002847852972237367,
 'SUBCAT Foreign Languages': -0.0017186680633591398,
 'SUBCAT Civics & Government': 0.0005140673296167404,
 'SUBCAT Parent Involvement': 0.0009905866100162147,
 'SUBCAT Financial Literacy': 0.0065718344250536445,
 'SUBCAT Nutrition Education': 0.0017995943835159597,
 'SUBCAT Community Service': 0.0009094645635812981,
 'SUBCAT Economics': 0.0012570439675198908,
 'RESCAT Supplies': -0.19139180814208856,
 'RESCAT Technology': -0.1740392376320493,
 'RESCAT Books': 0.07403893716480617,
 'RESCAT Computers & Tablets': 0.06301674205250818,
 'RESCAT Educational Kits & Games': 0.04563414984523414,
 'RESCAT Instructional Technology': 0.037265324102779496,
 'RESCAT Reading Nooks, Desks & Storage': 0.027151049040676063,
 'RESCAT Flexible Seating': 0.023798335313490418,
 'RESCAT Trips': 0.005797012257318246,
 'RESCAT Classroom Basics': 0.023010833488723004,
 'RESCAT Other': -0.009730279860252674,
 'RESCAT Art Supplies': 0.015995695050181603,
 'RESCAT Lab Equipment': 0.01568834613230089,
 'RESCAT Sports & Exercise Equipment': 0.017190207245374275,
 'RESCAT Food, Clothing & Hygiene': 0.012982815512523421,
 'RESCAT Musical Instruments': 0.010240349272989488,
 'RESCAT Visitors': 0.00030884354401461075,
 'student': 0.011999240835749853,
 'school': 0.02847537428042968,
 'learn': -0.03258168079200721,
 'classroom': -0.04419165658458639,
 'help': -0.016203675419747755,
 'work': -0.024830132949782317,
 'read': -0.08758572546595868,
 'love': 0.0038670017249353423,
 'day': 0.01864482146185271,
 'class': -0.010664263085390025,
 'skill': -0.004051605328362676,
 'book': 0.08891123239443433,
 'technology': -0.11808148897776176,
 'time': -0.03828046676455201,
 'one': -0.013698750028762858,
 'math': -0.03259576722352694,
 'material': -0.004415810549827926,
 'grade': -0.03598273405249703,
 'children': -0.034247693857368665,
 'different': -0.003268124289051787,
 'project': 0.025158522832961355,
 'teach': -0.033970627303085775,
 'like': -0.00347756367845725,
 'world': 0.03116739127851184,
 'create': 0.0016828536630393057,
 'best': 0.003262051110283812,
 'learners': 0.004210655545903511,
 'science': 0.031004027751371434,
 'education': -0.02150298612148186,
 'community': 0.038937585915636036,
 'language': 0.025340087778046357,
 'home': 0.012366785045436977,
 'activities': -0.03720693849990379,
 'free': 0.0036961382087381946,
 'access': -0.0036620869763228396,
 'opportunity': -0.003354965560059011,
 'life': 0.0020115152566479635,
 'first': 0.01722343165478454,
 'fun': -0.017475770193364368,
 'hard': 0.005316999701569931,
 'environment': -0.015014397830033352,
 'lunch': 0.013053854443090164,
 'resource': -0.014297770997661751,
 'experience': 0.02534696634867169,
 'opportunities': 0.014809217718014743,
 'excited': 0.016828821838213624,
 'diverse': 0.05852717296637586,
 'eager': -0.004577448377166036,
 'play': 0.05797370870760576,
 'art': -0.013101406079525585,
 'challenge': 0.03079511725692637,
 'creative': 0.003910213741442937,
 'goal': 0.009096277864894217,
 'music': 0.04180441724848595,
 'amazing': 0.01280515760706807,
 'social': 0.005537804932409498,
 'poverty': -0.009316069963444798,
 'games': 0.0045410344301653124,
 'hands-on': 0.007477850360871758,
 'research': -0.02901829895365344,
 'knowledge': -0.012215320220662433,
 'engaging': 0.01403594347559283,
 'safe': 0.013164625146149354,
 'computer': -0.022315610149847842,
 'literacy': 0.008574475625969856,
 'reduced': -0.01336023829928359,
 'however': -0.011229118773993931,
 'comfortable': -0.03274148709919751,
 'band': 0.01858247027019884,
 'instrument': 0.03176419463584191,
 'musical': 0.01069595718495234,
 'healthy': 0.007451734197098925,
 'breakfast': 0.016932998195825444,
 'hungry': 0.012336403239056796,
 'team': 0.05600358970159896,
 'sport': 0.0167304533199452,
 'hurricane': 0.06168902208641233,
 'health': -0.005649193766752872,
 'volleyball': 0.006319555881715688,
 'basketball': 0.01736761295049605,
 'soccer': 0.01102636374797383,
 'college': -0.001402934110751089,
 'museum': 0.003495619680884088,
 'paint': -0.0009725668333258118,
 'activity': -0.006963283625460165}

word_weights = {'student': 0.011999240835749853,
  'school': 0.02847537428042968,
  'learn': -0.03258168079200721,
  'classroom': -0.04419165658458639,
  'help': -0.016203675419747755,
  'work': -0.024830132949782317,
  'read': -0.08758572546595868,
  'love': 0.0038670017249353423,
  'day': 0.01864482146185271,
  'class': -0.010664263085390025,
  'skill': -0.004051605328362676,
  'time': -0.03828046676455201,
  'one': -0.013698750028762858,
  'material': -0.004415810549827926,
  'grade': -0.03598273405249703,
  'children': -0.034247693857368665,
  'different': -0.003268124289051787,
  'project': 0.025158522832961355,
  'teach': -0.033970627303085775,
  'like': -0.00347756367845725,
  'world': 0.03116739127851184,
  'create': 0.0016828536630393057,
  'best': 0.003262051110283812,
  'learners': 0.004210655545903511,
  'education': -0.02150298612148186,
  'home': 0.012366785045436977,
  'activities': -0.03720693849990379,
  'free': 0.0036961382087381946,
  'access': -0.0036620869763228396,
  'opportunity': -0.003354965560059011,
  'life': 0.0020115152566479635,
  'first': 0.01722343165478454,
  'fun': -0.017475770193364368,
  'hard': 0.005316999701569931,
  'lunch': 0.013053854443090164,
  'resource': -0.014297770997661751,
  'experience': 0.02534696634867169,
  'opportunities': 0.014809217718014743,
  'excited': 0.016828821838213624,
  'diverse': 0.05852717296637586,
  'eager': -0.004577448377166036,
  'play': 0.05797370870760576,
  'challenge': 0.03079511725692637,
  'creative': 0.003910213741442937,
  'goal': 0.009096277864894217,
  'amazing': 0.01280515760706807,
  'social': 0.005537804932409498,
  'poverty': -0.009316069963444798,
  'hands-on': 0.007477850360871758,
  'knowledge': -0.012215320220662433,
  'engaging': 0.01403594347559283,
  'safe': 0.013164625146149354,
  'reduced': -0.01336023829928359,
  'however': -0.011229118773993931,
  'comfortable': -0.03274148709919751,
  'healthy': 0.007451734197098925,
  'hungry': 0.012336403239056796,
  'team': 0.05600358970159896,
  'health': -0.005649193766752872,
  'activity': -0.006963283625460165}

sample = "Imagine having tables and having to stand to play and complete work activities. I  teach preschool special education in an urban school district.  Most of the students are on free and reduced lunch.  My students are between the ages of 3-5 years old.  My students have a variety of disabilities including Autism, Down Syndrome, and language impairments. With the chairs, my students will be able to sit down to play with table toys like Legos and stringing beads.  My students will also be able to sit to complete work activities such as cutting with scissors, tracing their names and completing letter and number crafts. Donations will help my children sit and concentrate on their work.  My students will be better prepared for kindergarten.  They will be able to sit and learn basic skills such as colors, shapes, numbers and letters."

# this piece of html and javascript is customized from https://stackoverflow.com/questions/304837/javascript-user-selection-highlighting
javascript_highlight_head = """
                <style type="text/css">
                    .highlight
                    {
                        background-color: yellow;
                    }
                    #test-text::-moz-selection { /* Code for Firefox */

                        background: yellow;
                    }

                    #test-text::selection {

                        background: yellow;
                    }

                </style>
"""
# this piece of html and javascript is customized from https://stackoverflow.com/questions/304837/javascript-user-selection-highlighting
javascript_highlight_script = """
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
                <script type="text/javascript">
                    mouseXPosition = 0;
                    $(document).ready(function () {

                        $("#test-text, #test-text1, #test-text2, #test-text3").mousedown(function (e1) {
                            mouseXPosition = e1.pageX;//register the mouse down position
                        });

                        $("#test-text, #test-text1, #test-text2, #test-text3").mouseup(function (e2) {
                            var highlighted = false;
                            var selection = window.getSelection();
                            var selectedText = selection.toString();
                            var startPoint = window.getSelection().getRangeAt(0).startOffset;
                            var endPoint = window.getSelection().getRangeAt(0).endOffset;
                            var anchorTag = selection.anchorNode.parentNode;
                            var focusTag = selection.focusNode.parentNode;
                            if ((e2.pageX - mouseXPosition) < 0) {
                                focusTag = selection.anchorNode.parentNode;
                                anchorTag = selection.focusNode.parentNode;
                            }
                            if (selectedText.length === (endPoint - startPoint)) {
                                highlighted = true;

                                if (anchorTag.className !== "highlight") {
                                    highlightSelection();
                                } else {
                                    var afterText = selectedText + "<span class = 'highlight'>" + anchorTag.innerHTML.substr(endPoint) + "</span>";
                                    anchorTag.innerHTML = anchorTag.innerHTML.substr(0, startPoint);
                                    anchorTag.insertAdjacentHTML('afterend', afterText);
                                }

                            }else{
                                if(anchorTag.className !== "highlight" && focusTag.className !== "highlight"){
                                    highlightSelection();
                                    highlighted = true;
                                }

                            }


                            if (anchorTag.className === "highlight" && focusTag.className === 'highlight' && !highlighted) {
                                highlighted = true;

                                var afterHtml = anchorTag.innerHTML.substr(startPoint);
                                var outerHtml = selectedText.substr(afterHtml.length, selectedText.length - endPoint - afterHtml.length);
                                var anchorInnerhtml = anchorTag.innerHTML.substr(0, startPoint);
                                var focusInnerHtml = focusTag.innerHTML.substr(endPoint);
                                var focusBeforeHtml = focusTag.innerHTML.substr(0, endPoint);
                                selection.deleteFromDocument();
                                anchorTag.innerHTML = anchorInnerhtml;
                                focusTag.innerHTml = focusInnerHtml;
                                var anchorafterHtml = afterHtml + outerHtml + focusBeforeHtml;
                                anchorTag.insertAdjacentHTML('afterend', anchorafterHtml);


                            }

                            if (anchorTag.className === "highlight" && !highlighted) {
                                highlighted = true;
                                var Innerhtml = anchorTag.innerHTML.substr(0, startPoint);
                                var afterHtml = anchorTag.innerHTML.substr(startPoint);
                                var outerHtml = selectedText.substr(afterHtml.length, selectedText.length);
                                selection.deleteFromDocument();
                                anchorTag.innerHTML = Innerhtml;
                                anchorTag.insertAdjacentHTML('afterend', afterHtml + outerHtml);
                             }

                            if (focusTag.className === 'highlight' && !highlighted) {
                                highlighted = true;
                                var beforeHtml = focusTag.innerHTML.substr(0, endPoint);
                                var outerHtml = selectedText.substr(0, selectedText.length - beforeHtml.length);
                                selection.deleteFromDocument();
                                focusTag.innerHTml = focusTag.innerHTML.substr(endPoint);
                                outerHtml += beforeHtml;
                                focusTag.insertAdjacentHTML('beforebegin', outerHtml );


                            }
                            if (!highlighted) {
                                highlightSelection();
                            }
                            $('.highlight').each(function(){
                                if($(this).html() == ''){
                                    $(this).remove();
                                }
                            });
                            selection.removeAllRanges();
                        });
                    });

                    function highlightSelection() {
                        var selection;

                        //Get the selected stuff
                        if (window.getSelection)
                            selection = window.getSelection();
                        else if (typeof document.selection != "undefined")
                            selection = document.selection;

                        //Get a the selected content, in a range object
                        var range = selection.getRangeAt(0);

                        //If the range spans some text, and inside a tag, set its css class.
                        if (range && !selection.isCollapsed) {
                            if (selection.anchorNode.parentNode == selection.focusNode.parentNode) {
                                var span = document.createElement('span');
                                span.className = 'highlight';
                                span.textContent = selection.toString();
                                selection.deleteFromDocument();
                                range.insertNode(span);
                            }
                        }
                    }

                </script>
"""

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
    draw_v5()
    draw_v6()
    draw_v7()
    return

def draw_model():
    # call other functions for model here
    st.markdown(
    """
        ## Does the Model Agree with You?
    """
    )

    model_user_choose_donate()

    st.markdown(
    """
        ## Predicting if Your Project can be Fully-funded
    """
    )

    model_proj_desc_interaction()

    return

####################### model sections  #######################

def model_proj_desc_interaction():
    st.markdown(
    """
        ### Please enter your project proposal here!
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
        if cost == 0:
            st.markdown("> Please enter project cost!")
        if subcat == []:
            st.markdown("> Please enter project category!")
        if rescat == []:
            st.markdown("> Please enter resource category!")
        if cost == 0 or subcat == [] or rescat == []:
            return

        # process the input to form a single-row dataframe for prediction
        df = pd.read_pickle("modeling/features.pkl")
        df = df.append(pd.Series(dtype=object), ignore_index = True)
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
            st.write("Our model predicts that this project **can be** fully funded")
        else:
            st.write("Our model predicts that this project **can not be** fully funded")
            st.write("Here are some suggestions on the key words to be included in your project proposal: ")
            st.markdown('**' + ', '.join(calc_10_words(description)[:10]) + '**')
            # st.write(clf.predict_proba(df))
        st.write("Here is how our model feels about your project proposal")
        highlight_essay(description)

def calc_10_words(description):
    L = re.findall(r"[^A-Za-z-']+|[A-Za-z-']+", description)
    stem_to_weight, word_to_stem = compute_lr_words(L)
    stem_exists = { val: True for _, val in word_to_stem.items()}
    words = []
    word_sorted_weights = pd.DataFrame.from_dict(word_weights, orient="index").sort_values(by=[0], ascending=False)
    word_sorted_weights = word_sorted_weights[word_sorted_weights[0] > 0]
    for index, row in word_sorted_weights.iterrows():
        # print(index, row[0])
        if index not in stem_exists:
            words.append(index)
    return words

def model_user_choose_donate():
    st.markdown(
    """
        ### Will you donate to this project?
    """
    )
    def card_information(title, subcat, rescat, cost):
        # this piece of html is customized from https://www.w3schools.com/howto/howto_css_column_cards.asp
        style = """
                %s
                <style>div.card{background-color:#d0f5f3;border-radius: 5px;box-shadow: 0 0 0 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>
                <style>div.container{padding: 6px 16px;}</style>
                <style>div.column{float: left; width: 23%%; padding: 0 5px;}</style>
                <style>div.row {margin: 0 -2.5px;}</style>
                <style>div.row:after{content: ""; display: table; clear: both;}</style>
        """ % javascript_highlight_head
        card = """
                %s
                <div class="row">
                    <div class="column"><div class="card">
                        <div class="container" id="test-text"><h3><b> %s </b></h3><p >%s</p></div>
                    </div></div>
                    <div class="column"><div class="card">
                        <div class="container" id="test-text1"><h3><b> %s </b></h3><p >%s</p></div>
                    </div></div>
                    <div class="column"><div class="card">
                        <div class="container" id="test-text2"><h3><b> %s </b></h3><p >%s</p></div>
                    </div></div>
                    <div class="column"><div class="card">
                        <div class="container" id="test-text3"><h3><b> %s </b></h3><p >%s</p></div>
                    </div></div>
                </div>
                %s
        """ % (style, "Title", title, "Category", subcat, "Resource", rescat, "Target", cost, javascript_highlight_script)
        components.html(card, height=113)

    def card_description(text, h):
        header = "Project Description"
        # this piece of html is customized from https://discuss.streamlit.io/t/can-i-wrap-st-info-or-other-elements-inside-custom-html/618
        to_write_html = """
                %s
                <style>div.card{background-color:#fac5ce;border-radius: 5px;box-shadow: 0 0 0 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>
                <style>div.container{padding: 2px 16px;}</style>
                <div class="card">
                    <div class="container" id="test-text"><h3><b> %s </b></h3><p >%s</p></div>
                </div>
            %s
        """ % (javascript_highlight_head, header, text, javascript_highlight_script)
        components.html(to_write_html, height=h)

    title1 = 'A Hungry Stomach Cannot Hear'
    essay1 = '"A hungry stomach cannot hear" This quote by Jean De La Fontaine sums up the dilemma the students face every day. My students represent a diverse population. The majority of them are from lower income, single parent homes.  Their life experiences go beyond what most think is normal for their age group. They have a strong desire to succeed and learn, but sometimes their struggles at home intrude on their ability to focus.  Our school is part of the “Leader in Me” program, based on Covey’s Seven Habits of Highly Successful People. My students need snacks such as Cheetos, animal crackers and popcorn Students will be given a snack midway through the day. This will help them focus and avoid the afternoon energy crash. In addition the afternoon snack is used as an opportunity for leadership.  Students are chosen for responsible roles such as; setting up and distributing the snacks, ensuring every student receives their snack, cleaning up. The snack will help calm the anxiety associated with being hungry.  Student behavior will be better. Focus on the lessons will be higher. By allowing the students to focus they have a better chance of reaching their goals. '
    subcat1 = 'Literacy, Other'
    rescat1 = 'Other'
    cost1 = '$509.89'
    fully_funded1 = True
    height1 = 320
    title2 = "There's An App For That! (Part 2)"
    essay2 = 'My first grade students have an enthusiasm for learning that is contagious! Unfortunately, our district does not have the financial resources to provide access to some of the most up to date technology. My students deserve to have access to the same types of technology as their peers. I teach first grade and know that our students are eager to learn but lack some much needed educational items. Approximately 75% of the students in our district live in poverty.\n\nWe are a Title I school and also participate in the free and reduced lunch program. All of our students are eligible to receive free breakfast and lunch each school day. As a teacher, I want to provide the best educational environment and the most relevant learning experiences possible. With your help, my students will have access to current technology. They will have the opportunity to interact with various learning apps and programs and to receive much needed support in both literacy and math. My students will have access to the Amazon Fire tablets during both literacy and math center rotations. Students in need of additional support or remediation will receive it. My students will have the opportunity to utilize the plethora of free learning apps and programs available in the Google store. Students will also be able to create presentations for class projects and to share this learning with their families in a unique way.\n\n    Donations to this project will provide my students with access to technology that they may otherwise not have the opportunity to experience. Your support will ensure that my curious and eager learners keep pace with their peers and have an additional learning tool at their fingertips. '
    subcat2 = 'Literacy, Mathematics'
    rescat2 = 'Technology'
    cost2 = '$396.72'
    fully_funded2 = False
    height2 = 400
    sample = ["A", "B"]
    sample_choice = st.selectbox('Case', sample)
    if sample_choice == "A":
        card_information(title1, subcat1, rescat1, cost1)
        card_description(essay1, height1)
    elif sample_choice == "B":
        card_information(title2, subcat2, rescat2, cost2)
        card_description(essay2, height2)

    # select donate
    col1, col2, col3, col4 = st.beta_columns(4)
    donate = col2.button("Donate")
    later = col3.button("Maybe Later")
    if donate or later:
        if donate:
            st.markdown("Thanks for your donation!")
        if later:
            st.markdown("Looking forward to your donation in the future!")
        if sample_choice == "A":
            st.markdown("This project proposal <b>has been fully funded</b>. Our model predicts that it <b>can be</b> fully funded.", unsafe_allow_html=True)
            highlight_subcategories(subcat1)
            highlight_resource_categories(rescat1)
            highlight_cost(cost1)
            highlight_essay(essay1)
        elif sample_choice == "B":
            st.markdown("This project proposal <b>has ended without being fully funded</b>. Our model predicts that it <b>cannot be</b> fully funded.", unsafe_allow_html=True)
            highlight_subcategories(subcat2)
            highlight_resource_categories(rescat2)
            highlight_cost(cost2)
            highlight_essay(essay2)

def annotation(body, label="", background="#ddd", color="#333", **style):
    """
    THIS FUNCTION IS TAKEN FROM https://github.com/tvst/st-annotated-text/blob/master/annotated_text/__init__.py
    """

    if "font_family" not in style:
        style["font_family"] = "sans-serif"

    return span(
        style=styles(
            background=background,
            border_radius=rem(0.33),
            color=color,
            padding=(rem(0.17), rem(0.67)),
            display="inline-flex",
            justify_content="center",
            align_items="center",
            **style,
        )
    )(
        body,
        span(
            style=styles(
                color=color,
                font_size=em(0.67),
                opacity=0.5,
                padding_left=rem(0.5),
                text_transform="uppercase",
                margin_bottom=px(-2),
            )
        )(label)
    )

def annotated_text(*args, **kwargs):
    """
    THIS FUNCTION IS TAKEN FROM https://github.com/tvst/st-annotated-text/blob/master/annotated_text/__init__.py
    """
    out = div(style=styles(
        font_family="sans-serif",
        line_height="1.5",
        font_size=px(16),
    ))

    for arg in args:
        if isinstance(arg, str):
            out(arg)

        elif isinstance(arg, HtmlElement):
            out(arg)

        elif isinstance(arg, tuple):
            out(annotation(*arg))

        else:
            raise Exception("Oh noes!")

    components.html(str(out), **kwargs)

def highlight_subcategories(subcats, height=35):
    to_annotate = ["Categories: "]
    subcats_L = subcats.split(", ")
    for i in range(len(subcats_L)):
        subcat = subcats_L[i]
        feat = "SUBCAT %s" % subcat
        w = 0
        if feat in lr_weights:
            w = lr_weights[feat]
        if w < 0:
            to_annotate.append((subcat, "", "rgba(250, 0, 0, %.2f)" % abs(w)))
        else:
            to_annotate.append((subcat, "", "rgba(0, 128, 0, %.2f)" % abs(w)))
        if i < len(subcats_L) - 1:
            to_annotate.append(", ")
    annotated_text(*to_annotate, height=height)

def highlight_resource_categories(rescats, height=35):
    to_annotate = ["Resource Categories: "]
    rescats_L = rescats.split(", ")
    for i in range(len(rescats_L)):
        rescat = rescats_L[i]
        feat = "RESCAT %s" % rescat
        w = 0
        if feat in lr_weights:
            w = lr_weights[feat]
        if w < 0:
            to_annotate.append((rescat, "", "rgba(250, 0, 0, %.2f)" % abs(w)))
        else:
            to_annotate.append((rescat, "", "rgba(0, 128, 0, %.2f)" % abs(w)))
        if i < len(rescats_L) - 1:
            to_annotate.append(", ")
    annotated_text(*to_annotate, height=height)

def highlight_cost(cost, height=35):
    feat = 'Project Cost'
    w = lr_weights[feat] * float(cost[1:])
    to_annotate = ["Cost: ", (cost, "", "rgba(250, 0, 0, %.2f)" % abs(w))]
    annotated_text(*to_annotate, height=height)

def compute_lr_words(words):
    stem_to_weight = {}
    word_to_stem = {}
    for word in words:
        w = 0
        if word.endswith('s'):
            if word in lr_weights:
                w = lr_weights[word]
                word_to_stem[word] = word
                stem_to_weight[word] = stem_to_weight.get(word, 0) + w
            elif word[:-1] in lr_weights:
                w = lr_weights[word[:-1]]
                word_to_stem[word] = word[:-1]
                stem_to_weight[word[:-1]] = stem_to_weight.get(word[:-1], 0) + w
        elif word.endswith('ing'):
            if word in lr_weights:
                w = lr_weights[word]
                word_to_stem[word] = word
                stem_to_weight[word] = stem_to_weight.get(word, 0) + w
            elif word[:-3] in lr_weights:
                w = lr_weights[word[:-3]]
                word_to_stem[word] = word[:-3]
                stem_to_weight[word[:-3]] = stem_to_weight.get(word[:-3], 0) + w
        else:
            if word in lr_weights:
                w = lr_weights[word]
                word_to_stem[word] = word
                stem_to_weight[word] = stem_to_weight.get(word, 0) + w
    return stem_to_weight, word_to_stem

def highlight_essay(essay, height=500):
    to_annotate = ["Project Description: "]
    L = re.findall(r"[^A-Za-z-']+|[A-Za-z-']+", essay)
    stem_to_weight, word_to_stem = compute_lr_words(L)
    for word in L:
        if word not in word_to_stem:
            to_annotate.append(word)
        else:
            w = stem_to_weight[word_to_stem[word]]
            if w < 0:
                to_annotate.append((word, "", "rgba(250, 0, 0, %.2f)" % abs(w)))
            else:
                to_annotate.append((word, "", "rgba(0, 128, 0, %.2f)" % abs(w)))
    annotated_text(*to_annotate, height=height, scrolling=True)


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
    v2 = alt.Chart(df).mark_circle().encode(
            x=alt.X('School State:N'),
            y=alt.Y('School Metro Type:N'),
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
    grade = grade[grade['Project Grade Level Category'] != 'unknown']
    # grade = grade[grade['Post year'] != 2018]
    resource = pd.read_csv("data/resource_by_year.csv")
    # resource = resource[resource['Post year'] != 2018]

    grade_selector = alt.selection_single(fields=['Project Grade Level Category'])
    resource_selector = alt.selection_single(fields=['Project Resource Category'])

    slider = alt.binding_range(min=2013, max=2018, step=1)
    select_year = alt.selection_single(name="Year", fields=['Post year'],
                                       bind=slider, init={'Post year': 2017})
    g_histo = alt.Chart(grade).mark_bar().encode(
        x= "Project Grade Level Category:N",
        y= "Project Cost:Q",
        color=alt.condition(grade_selector,
                        "Project Grade Level Category:N",
                        alt.value('lightgray')),
        tooltip=["Project Grade Level Category:N", "Project Cost:Q"]
    ).add_selection(
        select_year,
        grade_selector
    ).transform_filter(
        select_year
    ).properties(
        width=500,
        height=500
    )

    g_line = alt.Chart(grade).mark_line().encode(
        x = 'Post year:N',
        y = 'Project Cost:Q',
        color = alt.Color("Project Grade Level Category:N"),
        tooltip=["Project Cost:Q"]
    ).add_selection(
        grade_selector
    ).transform_filter(
        grade_selector
    ).properties(
        width=500,
        height=500
    )

    r_histo = alt.Chart(resource).mark_bar().encode(
        x= "Project Resource Category:N",
        y= "Project Cost:Q",
        color=alt.condition(resource_selector,
                        "Project Resource Category:N",
                        alt.value('lightgray')),
        tooltip=["Project Resource Category:N", "Project Cost:Q"]
    ).add_selection(
        select_year,
        resource_selector
    ).transform_filter(
        select_year
    ).properties(
        width=500,
        height=500
    )

    r_line = alt.Chart(resource).mark_line().encode(
        x = 'Post year:N',
        y = 'Project Cost:Q',
        color = alt.Color("Project Resource Category:N"),
        tooltip=["Project Resource Category:N", "Project Cost:Q"]
    ).add_selection(
        resource_selector
    ).transform_filter(
        resource_selector
    ).properties(
        width=500,
        height=500
    )

    st.write(g_histo | g_line)
    st.write(r_histo | r_line)
    return

def draw_v5():
    rate = pd.read_csv('data/successful_rate_year.csv')

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['Post year'], empty='none')

    line = alt.Chart(rate).mark_line().encode(
            alt.X('Post year:N'),
            alt.Y('Rate:Q', title='Rate(%)'),
            color = 'Project Current Status:N'
    )

    selectors = alt.Chart(rate).mark_point().encode(
        x='Post year:N',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Rate:Q', alt.value(' '), format='.2f')
    )

    rules = alt.Chart(rate).mark_rule(color='gray').encode(
        x='Post year:N',
    ).transform_filter(
        nearest
    )

    v5 = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=1000,
        height=500
    )

    st.write(v5)

def draw_v6():
    rate = pd.read_csv('data/successful_rate_grade.csv')
    v6 = alt.Chart(rate).mark_bar().encode(
        x = 'Project Grade Level Category:N',
        y=alt.Y('Rate:Q', stack="normalize"),
        color='Project Current Status:N',
        tooltip=["Project Grade Level Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=300,
        height=400
    )
    st.write(v6)

def draw_v7():
    before_rate = pd.read_csv('data/successful_rate_before_2017_resource.csv')
    after_rate = pd.read_csv('data/successful_rate_2017_resource.csv')

    before = alt.Chart(before_rate).mark_bar().encode(
        x = alt.X('Project Resource Category:N'),
        y=alt.Y('Rate:Q', stack="normalize"),
        color='Project Current Status:N',
        tooltip=["Project Resource Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=300,
        height=400
    )

    after = alt.Chart(after_rate).mark_bar().encode(
        x = 'Project Resource Category:N',
        y=alt.Y('Rate:Q', stack="normalize"),
        color='Project Current Status:N',
        tooltip=["Project Resource Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=300,
        height=400
    )

    st.write(before | after)
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

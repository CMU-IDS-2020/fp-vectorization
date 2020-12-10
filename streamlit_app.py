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

lr_weights_essay = {'X_essay_len': 0.00037009498147819186,
 'X_essay_?!': 0.022862047528095543,
 'X_essay_numbers': 0.01710425182793344}

lr_weights_category = { 'Literacy': -0.02675414594610817,
 'Mathematics': -0.024337049153145347,
 'Literature & Writing': -0.043951058135067674,
 'Special Needs': -0.02235234610771705,
 'Applied Sciences': 0.023120203927975543,
 'Health & Wellness': 0.002686796406848133,
 'Visual Arts': 0.008322275126736616,
 'Environmental Science': 0.01815032189678106,
 'Early Development': -0.0006137717434503105,
 'ESL': 0.0026233691221089054,
 'Health & Life Science': -6.0406607719211255e-05,
 'Music': 0.012449827463276822,
 'History & Geography': -0.0017641748773900465,
 'Character Education': 0.0030711481263678182,
 'College & Career Prep': -0.004840627317758782,
 'Other': -0.004180130093053809,
 'Gym & Fitness': 0.004951805930879193,
 'Performing Arts': 0.005897827810465053,
 'Team Sports': 0.011797297473147181,
 'Social Sciences': 0.0018802486224543718,
 'Care & Hunger': 0.023113709439820516,
 'Warmth': 0.023113709439820516,
 'Extracurricular': 0.0032671657579344095,
 'Foreign Languages': -0.001940472307504656,
 'Civics & Government': 0.0008100275337310177,
 'Parent Involvement': 0.0012725859345385441,
 'Financial Literacy': 0.00659104076281808,
 'Nutrition Education': 0.0017930669266562297,
 'Community Service': 0.0008775629945810779,
 'Economics': 0.0017047452070680818}

lr_weights_resource = { 'Supplies': -0.18223053875245432,
 'Technology': -0.16515633212956615,
 'Books': 0.07102715935779005,
 'Computers & Tablets': 0.059992721113569014,
 'Educational Kits & Games': 0.04255974206230025,
 'Instructional Technology': 0.034088254186318494,
 'Reading Nooks, Desks & Storage': 0.025321222778042492,
 'Flexible Seating': 0.023876527709445025,
 'Trips': 0.005731370794660587,
 'Classroom Basics': 0.021247359963883987,
 'Other': -0.00856904074453615,
 'Art Supplies': 0.01676966165597499,
 'Lab Equipment': 0.015751967725681156,
 'Sports & Exercise Equipment': 0.014479932469530426,
 'Food, Clothing & Hygiene': 0.012732738456745468,
 'Musical Instruments': 0.009404971517750391,
 'Visitors': -0.00016772328167669807}

lr_weights_word = { 'student': 0.015830006695597587,
 'school': 0.0386006382171662,
 'learn': -0.033720483737574934,
 'classroom': -0.03544470351848238,
 'help': -0.009186524816622913,
 'work': -0.031093587902273956,
 'read': -0.07807575544123652,
 'love': 0.00046212423870278004,
 'day': 0.020491516926485584,
 'class': -0.010396402933375064,
 'skill': -0.0037495535594597278,
 'book': 0.0767846281717963,
 'technology': -0.10727399545936897,
 'time': -0.037276025409045455,
 'one': -0.01148176729252409,
 'math': -0.02448056133273145,
 'material': -0.0004139395086164916,
 'grade': -0.035172407151551566,
 'children': -0.032112605033261854,
 'different': 0.0012004447456104384,
 'project': 0.030184483764775945,
 'teach': -0.0426859941804912,
 'like': -0.004190880843611827,
 'world': 0.02734080736665711,
 'create': 0.0021117743708813936,
 'best': 0.0015624359969566835,
 'learners': 0.0011100206075668655,
 'science': 0.030562220168185522,
 'education': -0.024556698506710835,
 'community': 0.03789618181531188,
 'language': 0.017660241437839168,
 'home': -0.00016328048666269172,
 'activities': -0.034248727142116436,
 'free': -0.00213634995082573,
 'access': -0.006309076922232798,
 'opportunity': -0.0018552438229765074,
 'life': 0.0036345876914418438,
 'first': 0.024243179009701476,
 'fun': -0.015478617039202091,
 'hard': 0.009283759659148485,
 'environment': -0.015718214112552756,
 'lunch': 0.010581099589040443,
 'resource': -0.017600491749165146,
 'experience': 0.01778776055452313,
 'opportunities': 0.013213689872708058,
 'excited': 0.010483342907890485,
 'diverse': 0.04828156903946251,
 'eager': -0.011551638122448086,
 'play': 0.04826922674708718,
 'art': -0.000748304979541739,
 'challenge': 0.02245777016123485,
 'creative': -0.0012473913589247322,
 'goal': 0.0011875675337204435,
 'music': 0.03971574567591645,
 'amazing': 0.009803198794909807,
 'social': 0.007311947589605967,
 'poverty': -0.008279877260178748,
 'games': -0.000595465356198027,
 'hands-on': 0.01084279013654092,
 'research': -0.02835950246656446,
 'knowledge': -0.01662608373289963,
 'engaging': 0.018028627424831824,
 'safe': 0.015020204573861313,
 'computer': -0.0095977114692029,
 'literacy': 0.006588961653983116,
 'reduced': -0.010731324201486899,
 'however': -0.014249778792810997,
 'comfortable': -0.0333614055998911,
 'band': 0.013347726967345824,
 'instrument': 0.028640155385215905,
 'musical': 0.009582394022856718,
 'healthy': 0.011651793058400549,
 'breakfast': 0.013681190684310628,
 'hungry': 0.012288837925334275,
 'team': 0.05087242919096718,
 'sport': 0.010842237675922076,
 'hurricane': 0.05561882729909601,
 'health': -0.005383323164823248,
 'volleyball': 0.006962429153669344,
 'basketball': 0.012175392794040315,
 'soccer': 0.009932368130717252,
 'college': 0.003188325444301269,
 'museum': 0.004981751487039096,
 'paint': 0.0002642947918454473,
 'activity': -0.004370217046162636}

lr_weights = {'X_essay_len': 0.00037009498147819186,
 'X_essay_?!': 0.022862047528095543,
 'X_essay_numbers': 0.01710425182793344,
 'Project Valid Time': 0.0028482231532322437,
 'Project Cost': -0.0009317929497469895,
 'Teacher Project Posted Sequence': 0.00529824019759347,
 'Project Year 2016': -0.24049512269794815,
 'Project Year 2017': -0.055228001403471,
 'SUBCAT Literacy': -0.02675414594610817,
 'SUBCAT Mathematics': -0.024337049153145347,
 'SUBCAT Literature & Writing': -0.043951058135067674,
 'SUBCAT Special Needs': -0.02235234610771705,
 'SUBCAT Applied Sciences': 0.023120203927975543,
 'SUBCAT Health & Wellness': 0.002686796406848133,
 'SUBCAT Visual Arts': 0.008322275126736616,
 'SUBCAT Environmental Science': 0.01815032189678106,
 'SUBCAT Early Development': -0.0006137717434503105,
 'SUBCAT ESL': 0.0026233691221089054,
 'SUBCAT Health & Life Science': -6.0406607719211255e-05,
 'SUBCAT Music': 0.012449827463276822,
 'SUBCAT History & Geography': -0.0017641748773900465,
 'SUBCAT Character Education': 0.0030711481263678182,
 'SUBCAT College & Career Prep': -0.004840627317758782,
 'SUBCAT Other': -0.004180130093053809,
 'SUBCAT Gym & Fitness': 0.004951805930879193,
 'SUBCAT Performing Arts': 0.005897827810465053,
 'SUBCAT Team Sports': 0.011797297473147181,
 'SUBCAT Social Sciences': 0.0018802486224543718,
 'SUBCAT Care & Hunger': 0.023113709439820516,
 'SUBCAT Warmth': 0.023113709439820516,
 'SUBCAT Extracurricular': 0.0032671657579344095,
 'SUBCAT Foreign Languages': -0.001940472307504656,
 'SUBCAT Civics & Government': 0.0008100275337310177,
 'SUBCAT Parent Involvement': 0.0012725859345385441,
 'SUBCAT Financial Literacy': 0.00659104076281808,
 'SUBCAT Nutrition Education': 0.0017930669266562297,
 'SUBCAT Community Service': 0.0008775629945810779,
 'SUBCAT Economics': 0.0017047452070680818,
 'RESCAT Supplies': -0.18223053875245432,
 'RESCAT Technology': -0.16515633212956615,
 'RESCAT Books': 0.07102715935779005,
 'RESCAT Computers & Tablets': 0.059992721113569014,
 'RESCAT Educational Kits & Games': 0.04255974206230025,
 'RESCAT Instructional Technology': 0.034088254186318494,
 'RESCAT Reading Nooks, Desks & Storage': 0.025321222778042492,
 'RESCAT Flexible Seating': 0.023876527709445025,
 'RESCAT Trips': 0.005731370794660587,
 'RESCAT Classroom Basics': 0.021247359963883987,
 'RESCAT Other': -0.00856904074453615,
 'RESCAT Art Supplies': 0.01676966165597499,
 'RESCAT Lab Equipment': 0.015751967725681156,
 'RESCAT Sports & Exercise Equipment': 0.014479932469530426,
 'RESCAT Food, Clothing & Hygiene': 0.012732738456745468,
 'RESCAT Musical Instruments': 0.009404971517750391,
 'RESCAT Visitors': -0.00016772328167669807,
 'student': 0.015830006695597587,
 'school': 0.0386006382171662,
 'learn': -0.033720483737574934,
 'classroom': -0.03544470351848238,
 'help': -0.009186524816622913,
 'work': -0.031093587902273956,
 'read': -0.07807575544123652,
 'love': 0.00046212423870278004,
 'day': 0.020491516926485584,
 'class': -0.010396402933375064,
 'skill': -0.0037495535594597278,
 'book': 0.0767846281717963,
 'technology': -0.10727399545936897,
 'time': -0.037276025409045455,
 'one': -0.01148176729252409,
 'math': -0.02448056133273145,
 'material': -0.0004139395086164916,
 'grade': -0.035172407151551566,
 'children': -0.032112605033261854,
 'different': 0.0012004447456104384,
 'project': 0.030184483764775945,
 'teach': -0.0426859941804912,
 'like': -0.004190880843611827,
 'world': 0.02734080736665711,
 'create': 0.0021117743708813936,
 'best': 0.0015624359969566835,
 'learners': 0.0011100206075668655,
 'science': 0.030562220168185522,
 'education': -0.024556698506710835,
 'community': 0.03789618181531188,
 'language': 0.017660241437839168,
 'home': -0.00016328048666269172,
 'activities': -0.034248727142116436,
 'free': -0.00213634995082573,
 'access': -0.006309076922232798,
 'opportunity': -0.0018552438229765074,
 'life': 0.0036345876914418438,
 'first': 0.024243179009701476,
 'fun': -0.015478617039202091,
 'hard': 0.009283759659148485,
 'environment': -0.015718214112552756,
 'lunch': 0.010581099589040443,
 'resource': -0.017600491749165146,
 'experience': 0.01778776055452313,
 'opportunities': 0.013213689872708058,
 'excited': 0.010483342907890485,
 'diverse': 0.04828156903946251,
 'eager': -0.011551638122448086,
 'play': 0.04826922674708718,
 'art': -0.000748304979541739,
 'challenge': 0.02245777016123485,
 'creative': -0.0012473913589247322,
 'goal': 0.0011875675337204435,
 'music': 0.03971574567591645,
 'amazing': 0.009803198794909807,
 'social': 0.007311947589605967,
 'poverty': -0.008279877260178748,
 'games': -0.000595465356198027,
 'hands-on': 0.01084279013654092,
 'research': -0.02835950246656446,
 'knowledge': -0.01662608373289963,
 'engaging': 0.018028627424831824,
 'safe': 0.015020204573861313,
 'computer': -0.0095977114692029,
 'literacy': 0.006588961653983116,
 'reduced': -0.010731324201486899,
 'however': -0.014249778792810997,
 'comfortable': -0.0333614055998911,
 'band': 0.013347726967345824,
 'instrument': 0.028640155385215905,
 'musical': 0.009582394022856718,
 'healthy': 0.011651793058400549,
 'breakfast': 0.013681190684310628,
 'hungry': 0.012288837925334275,
 'team': 0.05087242919096718,
 'sport': 0.010842237675922076,
 'hurricane': 0.05561882729909601,
 'health': -0.005383323164823248,
 'volleyball': 0.006962429153669344,
 'basketball': 0.012175392794040315,
 'soccer': 0.009932368130717252,
 'college': 0.003188325444301269,
 'museum': 0.004981751487039096,
 'paint': 0.0002642947918454473,
 'activity': -0.004370217046162636}

general_word_weights = {'student': 0.015830006695597587,
 'school': 0.0386006382171662,
 'learn': -0.033720483737574934,
 'classroom': -0.03544470351848238,
 'help': -0.009186524816622913,
 'work': -0.031093587902273956,
 'read': -0.07807575544123652,
 'love': 0.00046212423870278004,
 'day': 0.020491516926485584,
 'class': -0.010396402933375064,
 'skill': -0.0037495535594597278,
 'time': -0.037276025409045455,
 'one': -0.01148176729252409,
 'material': -0.0004139395086164916,
 'grade': -0.035172407151551566,
 'children': -0.032112605033261854,
 'different': 0.0012004447456104384,
 'project': 0.030184483764775945,
 'teach': -0.0426859941804912,
 'like': -0.004190880843611827,
 'world': 0.02734080736665711,
 'create': 0.0021117743708813936,
 'best': 0.0015624359969566835,
 'learners': 0.0011100206075668655,
 'education': -0.024556698506710835,
 'home': -0.00016328048666269172,
 'activities': -0.034248727142116436,
 'free': -0.00213634995082573,
 'access': -0.006309076922232798,
 'opportunity': -0.0018552438229765074,
 'life': 0.0036345876914418438,
 'first': 0.024243179009701476,
 'fun': -0.015478617039202091,
 'hard': 0.009283759659148485,
 'lunch': 0.010581099589040443,
 'resource': -0.017600491749165146,
 'experience': 0.01778776055452313,
 'opportunities': 0.013213689872708058,
 'excited': 0.010483342907890485,
 'diverse': 0.04828156903946251,
 'eager': -0.011551638122448086,
 'play': 0.04826922674708718,
 'challenge': 0.02245777016123485,
 'creative': -0.0012473913589247322,
 'goal': 0.0011875675337204435,
 'amazing': 0.009803198794909807,
 'social': 0.007311947589605967,
 'poverty': -0.008279877260178748,
 'hands-on': 0.01084279013654092,
 'knowledge': -0.01662608373289963,
 'engaging': 0.018028627424831824,
 'safe': 0.015020204573861313,
 'reduced': -0.010731324201486899,
 'however': -0.014249778792810997,
 'comfortable': -0.0333614055998911,
 'healthy': 0.011651793058400549,
 'hungry': 0.012288837925334275,
 'team': 0.05087242919096718,
 'health': -0.005383323164823248,
 'activity': -0.004370217046162636}

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
    """
        ## Background

        What happens when the school lacks funding? Based on the data
        collected by the Virginia Department of Education, students have
        fewer choices in courses, less experienced instructors, lower test
        scores and college enrollment in high poverty schools compared to
        low poverty ones. Even worse, according to the Center for American
        Progress, many school districts are impacted by the Covid-19 pandemic
        and will expect a higher budget loss in the coming years. Thus, financial
        support becomes increasingly important at this moment. Even though the
        funding itself could not be the panacea, an increasing amount of funding
        will offer students more educational opportunities, better supplies, as
        well as mental services.

        In order to relieve the stress and help more
        donors recognize the challenges public schools are currently facing,
        we use the datasets from [DonorsChoose.org](https://www.kaggle.com/donorschoose/io),
        a funding platform for public school teachers to find resources in need,
        to make visualizations. We would like to use our visualizations to present
        donors with useful information and attract more potential donors.

        In addition, observing that some of the project proposals hosted on the platform
        might not get fully funded after the expiration date, we would like to
        develop a model to predict whether a project can be fully-funded or not
        based on the project proposal content. We aim to build a tool for teachers
        to better understand what are some characteristics for projects to get
        fully-funded, and how should they refine their project proposals.
    """

def draw_narrative():
    # call other functions for narrative here
    st.markdown(
    """
        # Visualization
        In this section, we will demonstrate statistical analysis about
        the donation data in serveral dimensions via interactive visualizations.
        
        Note that we might see some values drop dramatically at the year of 2018,
        which is attributed to the incomplete data collected at that year. But we
        still keep these data, as they are still valuable to be compared
        horizontally with respect to other dimensions (e.g., states).

        ## 1. Donation Cost Analysis

        ### 1.1 How did project cost distribute geographically?
        Usage: The color of the map indicates the average cost of projects in 
        each state. The tooltip has extra information about the sum cost of 
        projects. Further, by clicking on a specific state, its average and 
        sum project costs are shown in the following line charts.

    """
    )
    draw_v1_modified()

    st.markdown(
    """
        The map visualization shows the donation request from each state. 
        The amount of donation increases as the color gets darker. One of the 
        lighter places like Indiana has a proposed donation mean at around 602 
        dollars, while one of the darkest places like Wyoming has a proposed 
        donation mean at around 874 dollars. Thus, we further investigate 
        external reasons causing the phenomenon. An intuitive thought is 
        that states that have less state education fundings tend to need more 
        fundings from platforms like DonorsChoose.org. However, the reality 
        was the opposite. Based on the U.S.News (2020), the report from 
        2016-2017 school year showed that Vermont, New York, New Jersey, 
        Pennsylvania and Wyoming were the top five states with the most 
        state funding per pupil. Back to our visualization, the mean costs of 
        donation requests from all the five states are above 760, except for 
        Pennsylvania. Since the states with relatively darker colors match with 
        the states who have high educational fundings, it is possible that the 
        states with higher average donation requests may put more emphasis on 
        education: even though they have plenty of state fundings for the public 
        schools, teachers and schools are still striving for getting better 
        resources for their students. 
        
        Futhermore, if we click on specific states, we could also see the trends of 
        donation requests through 2013-2018, which is also interesting to have 
        further exploration. For instance, if we take a look at the state of 
        Hawaii, we could see the change of its average donation request and 
        its total donation request from 2013 to 2018. These three visualizations 
        are giving donors an overview of how much money each state needs for 
        their public classroom projects.
        
        ## 1.2 How did requested cost distribute with respect to different categories?
        Usage: By clicking on a bar in the left char, its value change over the years will be demonstrated on the right chart.
    """
    )
    # draw_v1()
    draw_v4()
    st.markdown(
    """

    """
    )
    st.markdown(
    """
        Understanding learners’ needs is an important aspect for donors, and 
        we will discuss the needs based on grade level and resource category. 
        For the grade level bar chart, we could see that the trend stays the 
        same from the year 2013 to 2018, where grades Pre K-2 always maintain 
        the highest project costs, then come grades 3-5, grades 6-8, and grades 
        9-12. Since this chart is showing the project request costs, no matter 
        if the projects succeed or not, we could assume that lower grades need 
        more fundings compared to higher grades, as they may need more supplies 
        in the classroom.

        We will take a further exploration of learners’ needs from the resource 
        category chart. This is a similar chart showing the change of supplies 
        throughout different years. In 2013, the order from most needed resources 
        to least was technologies, supplies, books, others, trips, and visitors. T
        he trend stayed in 2014 and 2015, but in 2015, the needs for other, trips, 
        and supplies increased. In 2016, the need for supplies surged, and it 
        even surpassed the need for technology. In 2017, more categories have 
        been added, including instructional technology, lab experiments, computers 
        and tablets, educational kits and games, flexible seating, clothing and 
        hygiene, musical instruments, reading nooks, desks, storages, and sports 
        and exercise equipment. 2017 is a transition year, where we could see 
        how the emphasis of public schools changed. General supplies were split 
        into several detailed categories, like flexible seating, hygiene, and 
        reading nooks, and different types of educational technologies began 
        to rise in public classrooms. The phenomenon could be explained by 
        several reasons: increasing awareness of students’ physical and mental 
        health, and the rapid development of technology. Physical health is 
        always an important topic in schools, because students are more likely 
        to receive high academic achievements with healthy bodies (Pennsylvania 
        Department of Education). Similarly, as more mental crises are reported, 
        mental health weighs the same importance as physical health, and schools 
        start to pay extra attention to the stress, pressure and depression 
        problems students are facing. Thus, multi-tiered systems of support are 
        being used to intervene in students’ behaviors, and flexible seating is 
        one of the options that could help students to create comfortable and 
        safe environments in the classroom, and it counted as alternative seating 
        for students who have special needs. For educational technology, as the 
        National Education Association (NEA) supports, that “Every student needs 
        the ability to navigate through the 24/7 information flow that today 
        connects the global community.” Thus, many of the public schools 
        implemented the “laptop programs,” where whole classes are provided with 
        laptops so that teachers can experiment with online teaching and learning 
        processes. At the same time, GoogleClassrooms and other educational 
        platforms are introduced to support and motivate learning, and technology 
        gradually becomes an indispensable part of the classroom. And if we scroll 
        the graph to 2018, we could see that computers and tablets become the most 
        needed category among all the resources.

        ### 1.3 How much money did each state donate?

    """
    )
    draw_v3()
    st.markdown(
    """
        These two graphs show the mean and the sum of donation records. Based on the graphs, we could see that the mean donation for most of the states is around 40 to 70 dollars, with some outliers of Hawaii, North Dakota, and Idaho, where the mean donation of Hawaii reaches 100 dollars per donation. As for the sum record graph, California, New York and Texas have the most donations added together, where we can make an assumption that the people at these places put a higher emphasis on education as they are willing to donate for school projects. However, this sum may be affected by the population so there is still bias available.

        But based on these two graphs, potential donors can have brief ideas on how the donations went on in different states, and new donors can use the donation mean chart as their reference.
        ## 2. Successful Rate Analysis

        ### 2.1 Successful Rate Variation over Years
        In the successful rate section, we used a series of graphs revealing the current status of projects and the idea of what kind of projects may receive higher successful rates compared to the others.
    """
    )
    draw_v5()
    st.markdown(
    """
<<<<<<< HEAD
        In the line chart, the slopes for both fully funded and expired rates do not fluctuate much, where the fully funded rate keeps at around 72~78 percent. We did not include the dataset from 2018 in the line chart since it was not complete yet. We wish that the fully funded rates could elevate after donors take a glance over our visualizations and have a better understanding of learners’ needs at public schools. 
        ### 2.2 Successful Rate under Grade levels and Resource Categories
=======
        In the line chart, the slopes for both fully funded and expired rates do not fluctuate much, where the fully funded rate keeps at around 72~78 percent. We did not include the dataset from 2018 in the line chart since it was not complete yet. We wish that the fully funded rates could elevate after donors take a glance over our visualizations and have a better understanding of learners’ needs at public schools.
        ### Successful Rate under Grade levels and Resource Categories
>>>>>>> 897f9998d1e5de61146cd23a2b542c14d7196243
    """
    )
    draw_v6()
    draw_v7()
    st.markdown(
    """
        Based on the graph, the average successful rate for Pre K to grade 12 maintains at around 0.76. Thus, we can conclude that the grade level would not vitally affect the successful rate, and classroom teachers from each grade do not have to concern their grade level as an influential factor for a successful donation request.

        However, we could see obvious differences among different resources categories. From both of the graphs, technology/computer & tablets have the least successful rate compared to other categories. Connected to the graph below, where higher proposed costs may lead to lower successful rate, we believe that technology equipment including computers and tablets may have higher costs and may need future investments on maintenance and repair.

        ### 2.3 Successful Rate under Different School Metro Types and investigation about free lunch rates
    """
    )
    draw_v2()
    draw_v8()
    st.markdown(
    """
        According to the Federal Register, Child Nutrition Programs: Income 
        Eligibility Guidelines, schools are required to serve meals at no 
        charge to children whose household income is at or below 130 percent 
        of the Federal poverty guidelines. Children are entitled to pay a 
        reduced price if their household income is above 130 percent but at 
        or below 185 percent of these guidelines. In other words, when the 
        household income of a family reaches the poverty level or is below 
        the guideline, the kids in the family are eligible for reduced or free 
        lunch. Thus, the less average household income tax received by the 
        districts, the higher percentage of free lunch offered by the public 
        schools. In this way, we could assume that schools with higher free 
        lunch percentages will have a higher possibility of requiring funding 
        projects. Based on that, our visualization gives an overview of the 
        free lunch conditions in different school metro types within each 
        state. In general, we could suggest that schools in urban areas have a 
        higher percentage of school lunch, indicating higher needs for funding, 
        compared to those in rural, suburban and town. But there are some 
        exceptions, like Wyoming, Montana, and Idaho. In addition, even though 
        there is the bias that this visualization does not cover all the 
        schools in the United States, we could see some of the states have 
        darker bubbles than other states, like District of Columbia, Louisiana, 
        Mississippi, Illinois, New Jersey, Oklahoma, Pennsylvania, and Tennessee, 
        showing their greater percentage for free lunch, which could be 
        interpreted as a signal for funding. 

        From the second graph, we could see that urban school districts 
        have higher success rates than other metro types. With the conclusion 
        we got above, we infer that the amount of fundings the type of schools 
        need may affect their success rate. For example, urban schools who have 
        a higher percentage of free lunch rate may start more projects compared 
        to other school types, which may indirectly affect their success rate 
        (e.g., winning more favors of donors). 
        
        ### 2.4 Successful Rate Based on States
    """
    )
    draw_v9()
    st.markdown(
    """
        In this visualization, we are presenting the successful rate of 
        projects in each state. An interesting insight here is that there 
        are state clustering patterns reflected by the map: the southern part 
        of the U.S., especially the states in the southeast of the U.S. have 
        low successful rates in general, while the northern part has relatively 
        high rates. 
        ### 2.5 Successful Rate Based on Proposed Cost Intervals
    """
    )
    draw_v10()
    st.markdown(
    """
        We used another bar diagram to show the successful rate for different 
        cost intervals. Thus, both classrooms teachers and donors will know 
        where an ideal project cost should be at. Based on the graph, we could 
        see that the fully funded rates decrease as the costs increase when 
        the project cost is under 1000 dollars. However, the trend stops when 
        the cost is around 1000 dollars, where the successful rate maintains 
        at 0.5 when the cost is above 1000 dollars. This insight is a little 
        counterintuitive, because we thought the rate should always be linearly 
        decreasing with respect to the increase of the project cost. 
    """    
    )
    return

def draw_model():
    st.markdown(
    """
    # Model

    In the previous section, we explored what features of a project affect the fully funded rate. To help teachers formulate better project proposal, we build a machine learning model that predicts whether a project can get fully funded based on project proposal information.

    On this page, you can interact with our model.
    """
    )

    st.markdown(
    """
        ## Model Statistics
    """
    )

    model_display()

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

def model_display():
    st.markdown(
    """
        This model uses Logistic Regression with the following features:
        * length of the project description
        * number of '?' and '!' in the project description
        * number of numerical expressions in the project description, such as 2, 4, tweenty, one hundered
        * 85 high frequency words, such as student, learn, hungry, healthy
        * project target cost
        * project year
        * project valid duration (expiration date - start date)
        * project categories
        * resource categories

        We split the dataset into training set (80%) and test set (20%), and then we downsample the majority class to make the training dataset balance, and we calculated F1-score on the test set. The current f1-score is 0.77.

        We use all data from 2016 to 2018 to train our model. From our training result, we have discovered follwoing patterns:

    """
    )
    essay_df = pd.DataFrame.from_dict(lr_weights_essay, orient="index")
    essay_df['weight'] = essay_df. loc[:, 0]
    essay_df['category'] = essay_df.index.values

    category_df = pd.DataFrame.from_dict(lr_weights_category, orient="index")
    category_df['weight'] = category_df. loc[:, 0]
    category_df['category'] = category_df.index.values

    resource_df = pd.DataFrame.from_dict(lr_weights_resource, orient="index")
    resource_df['weight'] = resource_df. loc[:, 0]
    resource_df['category'] = resource_df.index.values

    word_df = pd.DataFrame.from_dict(lr_weights_word, orient="index")
    word_df['weight'] = word_df. loc[:, 0]
    word_df['category'] = word_df.index.values


    st.markdown(
    """
    ### Features from project description

    We have constructed several features based on the project description:

    * length of the project description
    * number of '?' and '!' in the project description
    * number of numerical expressions in the project
    * 85 high frequency words, such as student, learn, hungry, healthy

    Below are their corresponding weights learned by our model.
    """
    )

    chart = alt.Chart(essay_df).mark_bar().encode(
        x='weight',
        y=alt.Y('category',sort = '-x'),
        tooltip=['weight', 'category'],
        color=alt.condition(
            alt.datum.weight < 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).properties(width=600)
    st.write(chart)

    chart = alt.Chart(word_df).mark_bar().encode(
        y='weight',
        x=alt.X('category',sort = '-y'),
        tooltip=['weight', 'category'],
        color=alt.condition(
            alt.datum.weight < 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).properties(width=1000)
    st.write(chart)

    st.markdown(
    """
    ### Features from project categories

    We have computed one-hot encoding for each project category, to discover what are some project categories that introduce better chance of getting fully-funded.
    """
    )

    chart = alt.Chart(category_df).mark_bar().encode(
        x='weight',
        y=alt.Y('category',sort = '-x'),
        tooltip=['weight', 'category'],
        color=alt.condition(
            alt.datum.weight < 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).properties(width=600)
    st.write(chart)

    st.markdown(
    """
    ### Features from resource categories

    We have computed one-hot encoding for each resource category, to discover what are some resources that introduce better chance of getting fully-funded.
    """
    )

    chart = alt.Chart(resource_df).mark_bar().encode(
        x='weight',
        y=alt.Y('category',sort = '-x'),
        tooltip=['weight', 'category'],
        color=alt.condition(
            alt.datum.weight < 0,
            alt.value("steelblue"),  # The positive color
            alt.value("orange")  # The negative color
        )
    ).properties(width=600)
    st.write(chart)


    st.markdown(
    """
    """
    )

def model_proj_desc_interaction():
    st.info("You can enter your own projct proposal, and get predicted by our model that whether your project can get fully funded! After you have entered all needed information, please click on the button \"Submit Project Proposal\" to run our model!")

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

    submitted = st.button("Submit Project Proposal")
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
        # st.write("Here is how our model feels about your project proposal")
        st.info("Here we show how the model makes this prediction. The parts that encourage the model to predict fully-funded are highlighted in green, and the parts that discourages the model to predict fully-funded are highlighted in red. The darker the color, the more it influences the model.")
        highlight_essay(description)

def calc_10_words(description):
    L = re.findall(r"[^A-Za-z-']+|[A-Za-z-']+", description)
    stem_to_weight, word_to_stem = compute_lr_words(L)
    stem_exists = { val: True for _, val in word_to_stem.items()}
    words = []
    word_sorted_weights = pd.DataFrame.from_dict(general_word_weights, orient="index").sort_values(by=[0], ascending=False)
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
        """ % (style, "Title", title, "Project Category", subcat, "Resource Category", rescat, "Target", cost, javascript_highlight_script)
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
    title2 = "There's An App For That!"
    essay2 = 'My first grade students have an enthusiasm for learning that is contagious! Unfortunately, our district does not have the financial resources to provide access to some of the most up to date technology. My students deserve to have access to the same types of technology as their peers. I teach first grade and know that our students are eager to learn but lack some much needed educational items. Approximately 75% of the students in our district live in poverty.\n\nWe are a Title I school and also participate in the free and reduced lunch program. All of our students are eligible to receive free breakfast and lunch each school day. As a teacher, I want to provide the best educational environment and the most relevant learning experiences possible. With your help, my students will have access to current technology. They will have the opportunity to interact with various learning apps and programs and to receive much needed support in both literacy and math. My students will have access to the Amazon Fire tablets during both literacy and math center rotations. Students in need of additional support or remediation will receive it. My students will have the opportunity to utilize the plethora of free learning apps and programs available in the Google store. Students will also be able to create presentations for class projects and to share this learning with their families in a unique way.\n\n    Donations to this project will provide my students with access to technology that they may otherwise not have the opportunity to experience. Your support will ensure that my curious and eager learners keep pace with their peers and have an additional learning tool at their fingertips. '
    subcat2 = 'Literacy, Mathematics'
    rescat2 = 'Technology'
    cost2 = '$396.72'
    fully_funded2 = False
    height2 = 400
    sample = ["A", "B"]
    sample_choice = st.selectbox('Case', sample)
    st.info("As you read through the project description, you can highlight the text that incentivizes you to donate. After you are done reading and highlighting, please click on the button \"Donate\" or \"Maybe Later\" to tell us your preference and compare your decision with our model! ")
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
            st.info("Here we show how the model makes this prediction. The parts that encourage the model to predict fully-funded are highlighted in green, and the parts that discourages the model to predict fully-funded are highlighted in red. The darker the color, the more it influences the model.")
            highlight_subcategories(subcat1)
            highlight_resource_categories(rescat1)
            highlight_cost(cost1)
            highlight_essay(essay1)
        elif sample_choice == "B":
            st.markdown("This project proposal <b>has ended without being fully funded</b>. Our model predicts that it <b>cannot be</b> fully funded.", unsafe_allow_html=True)
            st.info("Here we show how the model makes this prediction. The parts that encourage the model to predict fully-funded are highlighted in green, and the parts that discourages the model to predict fully-funded are highlighted in red. The darker the color, the more it influences the model.")
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

def draw_v1_modified():
    avg_time_df = pd.read_csv("data/avg_loc_time_join.csv")
    avg_df = pd.read_csv("data/loc_cost_avg_join.csv")
    select_state = alt.selection_single(name="State",
                                        fields=['State'],
                                        init={'State': 'California'})

    states = alt.topo_feature(data.us_10m.url, 'states')

    highlight = alt.selection_single(on='mouseover', fields=['id'], empty='none')

    avg_v = alt.Chart(
        states
    ).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
        color = alt.Color('Mean Cost:Q',
                          scale=alt.Scale(scheme='greenblue')),
        tooltip=['State:N', 'Mean Cost:Q', 'Sum Cost:Q'],
        stroke=alt.condition(highlight, alt.value('red'), alt.value('black')),
        strokeWidth=alt.condition(highlight, alt.StrokeWidthValue(3), alt.StrokeWidthValue(1)),
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(avg_df, 'id', ['Mean Cost', 'Sum Cost', 'State'])
    ).add_selection(
        select_state,
        highlight
    ).project(
        type='albersUsa'
    ).properties(
        width=1000,
        height=700
    )

    avg_time_v = alt.Chart(avg_time_df).mark_area(
        color="lightblue",
        line=True
    ).encode(
        x = 'Post Year:N',
        y = 'Average:Q',
        tooltip=['Post Year:N', 'Average:Q']
    ).add_selection(
        select_state
    ).transform_filter(
        select_state
    ).properties(
        width=500,
        height=300
    )

    sum_time_v = alt.Chart(avg_time_df).mark_area(
        color="lightyellow",
        line=True
    ).encode(
        x = 'Post Year:N',
        y = 'Sum:Q',
        tooltip=['Post Year:N', 'Sum:Q']
    ).add_selection(
        select_state
    ).transform_filter(
        select_state
    ).properties(
        width=500,
        height=301
    )

    st.write(avg_v & (avg_time_v | sum_time_v))



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

    highlight = alt.selection_single(on='mouseover', fields=['id'], empty='none')

    count_v = alt.Chart(
        states
    ).mark_geoshape(
        stroke='gray',
        strokeWidth=1
    ).encode(
        color = 'Count:Q',
        tooltip=['State:N', 'Count:Q'],
        stroke=alt.condition(highlight, alt.value('red'), alt.value('gray')),
        strokeWidth=alt.condition(highlight, alt.StrokeWidthValue(2), alt.StrokeWidthValue(1)),
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(cnt_filter_df, 'id', ['Count', 'State'])
    ).add_selection(
        highlight
    ).project(
        type='albersUsa'
    ).properties(
        width=500,
        height=500
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
    slider = alt.binding_range(min=2013, max=2018, step=1, name='Year')
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

    slider = alt.binding_range(min=2013, max=2018, step=1, name='Year')
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
        y = 'Project Grade Level Category:N',
        x = alt.X('Rate:Q', stack="normalize"),
        order=alt.Order('Project Current Status:N', sort='descending'),
        color='Project Current Status:N',
        tooltip=["Project Grade Level Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=700,
        height=300
    )
    st.write(v6)

def draw_v7():
    before_rate = pd.read_csv('data/successful_rate_before_2017_resource.csv')
    after_rate = pd.read_csv('data/successful_rate_2017_resource.csv')

    before = alt.Chart(before_rate).mark_bar().encode(
        y = alt.Y('Project Resource Category:N'),
        x=alt.X('Rate:Q', stack="normalize"),
        order=alt.Order('Project Current Status:N', sort='descending'),
        color='Project Current Status:N',
        tooltip=["Project Resource Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=300,
        height=400
    )

    after = alt.Chart(after_rate).mark_bar().encode(
        y = 'Project Resource Category:N',
        x = alt.X('Rate:Q', stack="normalize"),
        color='Project Current Status:N',
        order=alt.Order('Project Current Status:N', sort='descending'),
        tooltip=["Project Resource Category:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=300,
        height=400
    )

    st.write(before | after)

def draw_v8():
    rate = pd.read_csv('data/successful_rate_metro.csv')
    v6 = alt.Chart(rate).mark_bar().encode(
        y = 'School Metro Type:N',
        x = alt.X('Rate:Q', stack="normalize"),
        order=alt.Order('Project Current Status:N', sort='descending'),
        color='Project Current Status:N',
        tooltip=["School Metro Type:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=700,
        height=300
    )
    st.write(v6)

def draw_v9():
    state_df = pd.read_csv("data/successful_rate_state.csv")

    highlight = alt.selection_single(on='mouseover', fields=['id'], empty='none')
    states = alt.topo_feature(data.us_10m.url, 'states')
    v_9 = alt.Chart(
        states
    ).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
        color = alt.Color('Fully Funded Rate:Q',
                          scale=alt.Scale(scheme='yelloworangebrown')),
        tooltip=['State:N', 'Fully Funded Rate:Q'],
        stroke=alt.condition(highlight, alt.value('red'), alt.value('black')),
        strokeWidth=alt.condition(highlight, alt.StrokeWidthValue(3), alt.StrokeWidthValue(1)),
    ).transform_lookup(
        lookup = 'id',
        from_ = alt.LookupData(state_df, 'id', ['Fully Funded Rate', 'State'])
    ).add_selection(
        highlight
    ).project(
        type='albersUsa'
    ).properties(
        width=1000,
        height=700
    )

    st.write(v_9)

def draw_v10():
    categoryNames = ['[0,100)', '[100,200)', '[100,200)',
                     '[200,300)', '[200,300)', '[300,400)',
                     '[300,400)', '[400,500)', '[400,500)',
                     '[500,600)', '[500,600)', '[600,700)',
                     '[600,700)', '[700,800)', '[700,800)',
                     '[800,900)', '[800,900)', '[900,1000)',
                     '[900,1000)', '[1000,2000)', '[1000,2000)',
                     '[2000,3000)', '[2000,3000)', '[3000,4000)',
                     '[3000,4000)', '[4000,5000)', '[4000,5000)',
                     '[5000,6000)', '[5000,6000)', '[6000,7000)',
                     '[6000,7000)', '[7000,8000)', '[7000,8000)',
                     '[8000,9000)', '[8000,9000)', '[9000,10000)',
                     '[9000,10000)', '[10000,inf)', '[10000,inf)']
    rate = pd.read_csv('data/successful_rate_cost_interval.csv')
    v6 = alt.Chart(rate).mark_bar().encode(
        y = alt.Y('Cost Interval:N',sort=categoryNames),
        x = alt.X('Rate:Q', stack="normalize"),
        order=alt.Order('Project Current Status:N', sort='descending'),
        color='Project Current Status:N',
        tooltip=["Cost Interval:N",
                  "Rate:Q",
                  "Project Current Status:N"]
    ).properties(
        width=700,
        height=500
    )
    st.write(v6)

    ####################### main #######################

sections = {
    'Description': draw_title,
    'Visualization': draw_narrative,
    'Model': draw_model,
}

st.set_page_config(layout="wide")

option = st.sidebar.selectbox(
    "Sections", list(sections.keys())
)
sections[option]()

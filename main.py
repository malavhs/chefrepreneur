import streamlit as st
import modules.restaurant_gen as restaurant_gen
import base64
from modules import vision_pipeline
from langchain.output_parsers import CommaSeparatedListOutputParser
from utils import prompts
num_sections = 3
st.set_page_config(layout="wide")

### INTRO
section1, section2, section3 = st.columns(num_sections)
with section1:
    st.write(' ')
with section2:
    st.title(":blue[Chef-repreneur]")
with section3:
    st.write(' ')
st.text("An AI tool to help a chef who's also an entrepreneur, looking to start his own business.")
st.markdown('''
            * Generates a recommended restaurant name and logo design based on selections.
            * Generates a signature dish based on available ingredients (uploaded via image)
            * Lists competitors in the city (using RAG)
            ''')

### INPUTS
with st.sidebar:
    form = st.form(key='rest-form', clear_on_submit=True)
    uploader = form.file_uploader('Ingredient Image')
    cuisine_picker = form.selectbox("Pick a cuisine", ("Mediterranean","Italian", "Indian", "Mexican", "Thai"))
    cost_picker = form.selectbox("Affordability Range",("High", "Low"))
    submit = form.form_submit_button('Submit')

### GENERATED CONTENT
st.divider()
box1, box2, box3= st.columns(num_sections)
if submit:
    with box1:
        if uploader is not None:
            text_gen = vision_pipeline.image_to_text(uploader, prompts.vision_prompt)
            text_summarized = vision_pipeline.text_to_text(text_gen, prompts.signature_dish_text_prompt)
            st.subheader('Signature Dish, powered by GPT4V & DALL E3')
            st.text(text_summarized)
            ai_image = vision_pipeline.text_to_image(text_summarized, prompts.signature_dish_image_prompt)
            st.image(ai_image)
    with box2:
        res_name = restaurant_gen.restaurant_name_generator(cuisine_picker, cost_picker)
        st.subheader('Restaurant Name & Logo, powered by DALL E3')
        st.text(res_name)
        res_logo = vision_pipeline.text_to_image(res_name, prompts.res_logo_prompt)
        st.image(res_logo)
    with box3:
        res_list = restaurant_gen.existing_names_list(cuisine_picker, cost_picker)
        st.subheader('Competitor Restaurants, powered by RAG techniques')
        output_parser = CommaSeparatedListOutputParser()
        res_list = output_parser.parse(res_list)
        for i in res_list:
            st.write(i)

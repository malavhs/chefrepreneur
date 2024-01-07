import streamlit as st
import modules.restaurant_gen as restaurant_gen
import base64
from modules import vision_pipeline
from utils import prompts
dataset = 'datasets/Zomato_Mumbai_Dataset_clean.csv'
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.title(":red[Chef-repreneur]")
with col3:
    st.write(' ')
st.text("An AI tool to help a chef who's also an entrepreneur, looking to start his own business.")
st.markdown('''
            * Generates a recommended restaurant name and logo design based on selections.
            * Generates a signature dish based on available ingredients (uploaded via image)
            * Lists competitors in the city (using RAG)
            ''')
if "expander_state" not in st.session_state:
    st.session_state["expander_state"] = True

def toggle_closed():
    st.session_state["expander_state"] = False

with st.expander("Make your selections",expanded = st.session_state["expander_state"]):
    form = st.form(key='rest-form', clear_on_submit=True)
    uploader = form.file_uploader('INGREDIENT IMAGE')
    cuisine_picker = form.selectbox(
        "Pick a cuisine",
        ("Italian", "Indian", "Mexican", "Thai")
    )
    cost_picker = form.selectbox(
        "Pick cost type",
        ("High", "Low")
    )
    submit = form.form_submit_button('Submit', on_click=toggle_closed)

c1, c2, c3= st.columns(3)
if submit:
    with c1:
        if uploader is not None:
            
            text_1 = vision_pipeline.image_to_text(uploader, prompts.vision_prompt)
            text_summarized = vision_pipeline.text_to_text(text_1, prompts.signature_dish_text_prompt)
            st.subheader('AI Generated Signature Dish')
            st.write(text_summarized)
            ai_image = vision_pipeline.text_to_image(text_summarized, prompts.signature_dish_image_prompt)
            st.image(ai_image)
    with c2:
        res_name = restaurant_gen.restaurant_name_generator(cuisine_picker, cost_picker)
        st.subheader('AI Generated Restaurant Name with Logo')
        st.write(res_name)
        res_logo = vision_pipeline.text_to_image(res_name, prompts.res_logo_prompt)
        st.image(res_logo)
    with c3:
        res_list = restaurant_gen.existing_names_list(cuisine_picker, cost_picker)
        st.subheader('List of Existing Restaurants (via RAG)')
        res_list = output_parser.parse(res_list)
        for i in res_list:
            st.write(i)

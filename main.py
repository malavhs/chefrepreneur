import streamlit as st
import langchain_helper

# config
pinecone_env = 'gcp-starter'
pinecone_index_name = 'restaurant-project'
dataset = 'datasets/Zomato_Mumbai_Dataset_clean.csv'

# this is for storing data into pinecone for the first time
#data = chunk_data(dataset, 100, 20)
#index = store_vectors(data, pinecone_env, pinecone_index_name)

cuisine_picker = st.sidebar.selectbox(
    "Pick a cuisine",
    ("Italian", "Indian", "Mexican", "Thai")
)
cost_picker = st.sidebar.selectbox(
    "Pick cost type",
    ("High", "Low")
)

restaurant_names = langchain_helper.generate_names(cuisine_picker, cost_picker)
res_list = restaurant_names.strip().split(",")
st.header("Mumbai Restaurants (Langchain + Pinecone for RAG + Streamlit)")
print(res_list)
st.write("**Names**")
for i in res_list:
    st.write(i)



import streamlit as st

with st.sidebar:
    st.image("img.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profilling", "ML", "Download"])
    st.info("Application")

from importer import importer
from profilling import render_profiling_page
from machine_learning import render_ml_page
if choice == "Upload":
    importer()

elif choice == "Profilling":
    render_profiling_page()

elif choice == "ML":
    render_ml_page()

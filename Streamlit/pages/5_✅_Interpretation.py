import streamlit as st 

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="✅"
)

st.title("Interprétation des résultats")










# ------------------------------------------------------------------------------------------------
# CSS 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("Streamlit/styles.css")
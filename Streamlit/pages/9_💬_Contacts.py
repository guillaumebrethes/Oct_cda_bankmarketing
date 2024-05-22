import streamlit as st  # type: ignore

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="💬"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

st.markdown('<h1 class="custom-title">Contacts</h1>', unsafe_allow_html=True)

if st.button("◀️\u2003 🎲 Avez vous souscrit ?"):
    st.switch_page("pages/7_🎲_Avez_vous_souscrit?.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


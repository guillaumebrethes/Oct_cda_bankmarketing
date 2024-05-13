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

if st.button("◀️\u20036 🎯 Recommandation_métier - Conclusion"):
    st.switch_page("pages/6_🎯_Recommandation_métier_-_Conclusion.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


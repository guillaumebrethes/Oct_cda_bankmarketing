import streamlit as st  # type: ignore

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🔍" 
)

st.title("Exploration du jeu de données")

if st.button("◀️\u2003📖 Présentation - Exploration"):
    st.switch_page("pages/2_📖_Presentation_des_données.py")
st.write("---")

st.markdown("Dans ce chapitre nous allons etudier plus en profondeur notre jeux de données.\n\n Nous allons aborder l'étude selon 2 axes principaux :")

st.write(
    "- **La visualisation** à l'aide de graphique pertinant\n\n"
    "- **L'étude statistique** pour cohoborer notre exploration et visualisation"
    )

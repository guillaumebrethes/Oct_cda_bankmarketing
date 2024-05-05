import streamlit as st  # type: ignore

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🏠",
    layout="wide" 
)

st.title("Contexte du projet")
st.write("---")

st.write(
    "Ce projet s'inscrit dans le cadre de l'utilisation des sciences des données appliquées dans les entreprises de service et plus précisément dans le domaine bancaire.\n\n Au sein du secteur bancaire,l'optimisation du ciblage du télémarketing est un enjeu clé, sous la pression croissante d'augmenter les profits et de réduire les coûts.\n\n Nous avons à disposition les données de la dernière campagne télémarketing d'une banque pour la vente de dépôts à terme. Ce jeu de données est en accès libre sur la plateforme Kaggle.com.\n\n L'objectif est de prédire quels clients sont les plus susceptibles de souscrire au dépôt à terme."
)

url_image_contexte = "Streamlit/pages/banque.jpg"
st.image(url_image_contexte,
         width=500, 
         use_column_width='always', 
         output_format='auto')

# ------------------------------------------------------------------------------------------------
# bouton de basculement vers page suivante 
st.write("---")
if st.button("▶️\u2003📖 Présentation - Exploration"):
    st.switch_page("pages/2_📖_Presentation_-_Exploration.py")
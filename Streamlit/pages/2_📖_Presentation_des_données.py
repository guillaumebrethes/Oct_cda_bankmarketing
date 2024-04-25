import streamlit as st 
import pandas as pd

# Variables 
df = pd.read_csv("bank.csv")

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="📖" 
)

st.title("Présentation des données")
st.write("---")

# Texte introductif
st.write(
    "Nous avons un jeu de données qui se compose de **11 162 lignes et 17 colonnes**, il contient des "
    "valeurs **numériques** ainsi que des valeurs textuelles."
    )
st.write(
    "Dans ce jeu de données nous avons des informations sur les : "
    )
st.write(
    "- caractéristiques **socio-démographiques** "
    "*(âge, type d'emploi, situation familiale, niveau d'études)* "
    )
st.write(
    "- caractéristiques **bancaires** des clients " 
    "*(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)*"
    )
st.write(
    "- caractéristiques de la campagne tel que *(Durée du dernier appel, nombre de contacts avant la campagne*" 
    "*(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)*"
    )
st.write(
         "Vous pouvez visualiser les premières lignes de celui-ci ci-dessous"
         )



# Définir une clé unique pour la case à cocher
checkbox_key = "dataset_content"

# Afficher la case à cocher avec une apparence personnalisée
if st.checkbox(
    label="Contenu du Dataset", 
    key=checkbox_key, 
    help="Cliquez pour afficher le contenu du dataset"):
        st.dataframe(df.head())

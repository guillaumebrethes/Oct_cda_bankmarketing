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
    
    # Sélection numérique pour choisir le nombre de lignes à afficher
    num_rows = st.number_input(
        label="Nombre de lignes à afficher",
        help="Maximun 50 lignes",
        min_value= 1,  
        max_value= 50,  
        step= 1,  # Incrément
        value= 5  # défaut
    )

    # Afficher les premières 'num_rows' lignes du DataFrame
    st.dataframe(df.head(num_rows))
    
    # Diviser l'espace en deux colonnes
    col1, col2 = st.columns(2)
    
    # Premier widget de sélection numérique pour choisir le numéro de la première ligne
    with col1:
        start_row = st.number_input(
            label="Première ligne à afficher",
            min_value = 0,  
            max_value = len(df) - 1, 
            step= 1, 
            value= 0  
        ) 
    
    # Deuxième widget de sélection numérique pour choisir le numéro de la dernière ligne
    with col2:
        end_row = st.number_input(
            label="Dernière ligne à afficher",
            min_value=start_row,  # Valeur minimale autorisée
            max_value=len(df),  # Nombre maximum de lignes du DataFrame
            step=1,  # Incrément
            value=min(start_row + 5, len(df))  # Valeur par défaut
        )

    # Afficher les lignes sélectionnées du DataFrame
    st.dataframe(df.iloc[start_row:end_row])


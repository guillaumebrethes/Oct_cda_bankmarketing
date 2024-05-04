import streamlit as st 
import pandas as pd
import time

# Variables 
df = pd.read_csv("/Users/gub/Documents/Privé/Formations/DataScientest/Data_projet/Oct_cda_bankmarketing/Streamlit/pages/bank.csv")

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="📖" 
)

st.title("Présentation des données")

# bouton de basculement vers page précédente
if st.button("◀️\u2003🏠 Contexte"):
    st.switch_page("pages/1_🏠_Contexte.py")
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
    "- caractéristiques de la campagne tel que *(Durée du dernier appel, nombre de contacts avant la campagne* " 
    "*(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)*"
    )
st.write(
         "Vous pouvez visualiser les lignes que vous désirez ci-dessous:"
         )

# ------------------------------------------------------------------------------------------------
# Afficher le conteneur expansible 
with st.expander(label="Contenu du Dataset", expanded=False):
    
    # Diviser l'espace en deux colonnes
    col1, col2 = st.columns(2)
    
    # Premier widget de sélection numérique pour choisir le numéro de la première ligne
    with col1:
        start_row = st.number_input(
            label = "Première ligne à afficher",
            min_value = 0,  # Valeur minimale autorisée
            max_value = len(df) - 1,  # Nombre maximum de lignes du DataFrame
            step = 1,  # Incrément
            value = 0  # Valeur par défaut
        )
    
    # Deuxième widget de sélection numérique pour choisir le numéro de la dernière ligne
    with col2:
        end_row = st.number_input(
            label = "Dernière ligne à afficher",
            min_value = start_row,  # Valeur minimale autorisée
            max_value = len(df),  # Nombre maximum de lignes du DataFrame
            step = 1,  # Incrément
            value = min(start_row + 5, len(df))  # Valeur par défaut
        )

    # Afficher les lignes sélectionnées du DataFrame
    selected_df = df.iloc[start_row:end_row]
    st.dataframe(selected_df)
    
# ------------------------------------------------------------------------------------------------
# Affichage des doublons 
st.write("---")
st.write(
         "Vous pouvez  les doublons et les valeurs manquantes ci-dessous:"
         )

# Diviser la page en deux colonnes
col1, col2 = st.columns(2)

# Expander pour les doublons
with col1.expander(label="Afficher les doublons", expanded=False):
    st.write(df.duplicated().sum())
    st.write("   ")

# Expander pour les valeurs manquantes
with col2.expander(label="Afficher les valeurs manquantes", expanded=False):
    st.dataframe(df.isna().sum())

# ------------------------------------------------------------------------------------------------
# Variable cible 
st.write("---")
url_page_exploration = "Streamlit/pages/3_🔍_Exploration.py"
st.markdown("La variable cible **`deposit`** est une valeur boolééne, qui réprésente la validation `1` ou non `0` du client du produit bancaire appelé **dépôt à terme**. \n\n Ce produit est souscrit\npar le client qui dépose une somme d'argent à la banque, qui sera bloquée sur une période données générant des intérets. Dans le jeu de\ndonnées elle se répartie en deux valeurs **'Yes et No'**.\n\n La page *Exploration des données* se concentre sur l'explotation des données dans leur ensemble.")

# ------------------------------------------------------------------------------------------------
# Tableau explicatif des variables  
st.write("---")
st.markdown("Vous avez la possibilité d'afficher les variables préssente dans notre jeux de données avec leur description et leur type.")

# bouton de basculement de page 
if st.button("▶️\u2003🔍 Exploration des données"):
    st.switch_page("pages/3_🔍_Exploration.py")

import streamlit as st  # type: ignore
import pandas as pd # type: ignore
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
         "Vous pouvez afficher les doublons et les valeurs manquantes ci-dessous:"
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
st.markdown("La variable cible **`deposit`** est une valeur boolééne, qui réprésente la validation `1` ou non `0` du client du produit bancaire appelé **dépôt à terme**. \n\n Ce produit est souscrit\npar le client qui dépose une somme d'argent à la banque, qui sera bloquée sur une période données générant des intérets. Dans le jeu de\ndonnées elle se répartie en deux valeurs **'Yes et No'**.\n\n La page *Exploration des données* se concentre sur l'explotation des données dans leur ensemble.")

# ------------------------------------------------------------------------------------------------
# Tableau explicatif des variables  
st.write("---")
st.markdown("Vous avez la possibilité d'afficher les variables préssente dans notre jeux de données avec leur description et leur type.")

with st.expander(label="Afficher le tableau des variables", expanded=False):
    st.markdown("### Tableau des variables")

    table_description = """
        | Colonne    | Description                                                              | Qualitative / Quantitative | Type     |
        |------------|--------------------------------------------------------------------------|----------------------------|----------|
        | age        | Âge du client                                                            |Quantitative - catégorielles|int64     |
        | job        | Type d'emploi du client                                                  |Qualitative - continues     |object    |
        | marital    | Statut marital du client                                                 |Qualitative - continues     |object    |
        | education  | Niveau d'éducation du client                                             |Qualitative - continues     |object    |
        | default    | Le client à t'il un défaut sur un crédit                                 |Qualitative - continues     |object    |
        | balance    | Solde moyen annuel sur le compte                                         |Quantitative - catégorielles|int64     |
        | housing    | Le client à t'il un prêt immobilier                                      |Qualitative - continues     |object    |
        | loan       | Le client à un prêt personnel en cour personnel                          |Qualitative - continues     |object    |
        | contact    | Type de communication pour contacter le client                           |Qualitative - continues     |object    |
        | day        | Jour du mois pour le dernier contact                                     |Qualitative - continues     |int64     |
        | month      | Mois de la dernière communication                                        |Qualitative - continues     |object    |
        | duration   | Durée de la dernière communication en secondes                           |Quantitative - catégorielles|int64     |
        | campaign   | Nombre de contacts effectués lors de cette campagne                      |Quantitative - catégorielles|int64     |
        | pdays      | Nombre de jours écoulés depuis le dernier contact de la dernière campagne|Quantitative - catégorielles|int64     |
        | previous   | Nombre de contacts effectués avant cette campagne                        |Quantitative - catégorielles|int64     |
        | poutcome   | Résultat de la dernière campagne de marketing précédente                 |Qualitative - continues     |object    |
        | deposit    | Le client à t'il souscrit à un dépôt à terme                             |Qualitative - continues     |object    |
        """

    st.markdown(table_description, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# Gestion des valeurs manquantes
st.write("---")
st.markdown("L'exploration des données nous a permis d'identifer que nous n'avons pas de valeur manquante. Par contre nous possèdons des valeurs qui ne nous parraisent pas exploitable en l'état")


with st.expander(label="Afficher le tableau des valeurs non désirés", expanded=False):
    st.markdown("### Tableau des valeurs non désirés")

    table_unknown = """
    | Variables | modalité en % | Modalité | Description de cette modalité    |
    |-----------|------------------|----------|-------------------------------|
    | job       | 1                | unknown  | Le "job" du client est inconnu |
    | education | 4                | unknown  | L'éducation du client est inconnue |
    | pdays     | 75               | -1       | La valeur -1 est attribuée aux nouveaux clients, c’est-à-dire aux clients qui n’ont jamais été appelés     pour une précédente campagne |
    | previous  | 75               | 0        | Comme les clients n'ont jamais été appelés suite à la précédente campagne, previous est égal à 0 |
    | poutcome  | 75               | unknown  | Ces mêmes nouveaux clients n'avaient jamais participé à une précédente campagne |
    | poutcome  | 5                | other    | |
    | contact   | 21               | unknown  | 21 % du moyen de contact est inconnu |
    """

    st.markdown(table_unknown, unsafe_allow_html=True)

st.markdown("Nous allons donc procéder à la suppression des **«unknown»** des modalités des varibles `job` et `education`, car elles représentent un volume minime de notre jeu de données (respectivement 1% et 4%).\n\n Les variables `pdays` et `previous` decrivent la même chose, nous décidons donc de garder qu’une seule variable `pdays`. Cette dernière nous apporte une information en plus; le nombre de jours écoulés depuis le dernier contact. \n\n Pour la variable `poutcome`, nous décidons de regrouper les 2 modalités unknown et other sous une même modalité commune (unknown), car il se peut que cela nous apporte une information supplémentaire lors de nos futures exploitations.\n\n La variable `contact` qui a un grand nombre d’inconnues n’a aucun enjeux métier, elle est donc supprimée.")












# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("---")
if st.button("▶️\u2003📊 Visualiation - Statistique"):
    st.switch_page("pages/3_📊_Visualiation_-_Statistique.py")
    


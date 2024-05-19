import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import time

# Variables 
df = pd.read_csv("bank.csv")
df_table_description = pd.read_csv("table_description.csv")
df_tableau_des_valeurs_non_désirées = pd.read_csv('Tableau_des_valeurs_non_désirées.csv')

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="📖"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

# titre 
st.markdown('<h1 class="custom-title">Présentation des données</h1>', unsafe_allow_html=True)


# bouton de basculement vers page précédente
if st.button("◀️\u2003🏠 Contexte"):
    st.switch_page("🏠_Projet_BankMarketing.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)

# Texte introductif
st.markdown(
    """
    Nous avons un jeu de données qui se compose de **11 162 lignes et 17 colonnes**, il contient des valeurs **numériques** ainsi que des valeurs textuelles.
    Dans ce jeu de données nous avons des informations sur les :
    - caractéristiques **socio-démographiques**
    
        *(âge, type d'emploi, situation familiale, niveau d'études)*
    - caractéristiques **bancaires** des clients 
        
        *(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)*
    - caractéristiques de la campagne tel que *(Durée du dernier appel, nombre de contacts avant la campagne* 
    
        *(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)*
    
    Vous pouvez visualiser les lignes que vous désirez ci-dessous:
         """)

# ------------------------------------------------------------------------------------------------
# Afficher le conteneur expansible 
with st.expander(label="Contenu du Dataset", 
                 expanded=False):
    
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
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
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
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown(
    """
    La variable cible <span class="orange-bold">deposit</span> est une valeur boolééne, qui réprésente la validation <span class="orange-bold">1</span> ou non <span class="orange-bold">0</span> du client du produit bancaire appelé <span class="orange-bold">dépôt à terme</span>. 
    
    Ce produit est souscrit par le client qui dépose une somme d'argent à la banque, qui sera bloquée sur une période données générant des intérets. 
    
    Dans le jeu de données elle se répartie en deux valeurs <span class="orange-bold">Yes</span> et <span class="orange-bold">No</span>. La page *Exploration des données* se concentre sur l'explotation des données dans leur ensemble.""",unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# Tableau explicatif des variables  
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("Vous avez la possibilité d'afficher les variables préssente dans notre jeux de données avec leur description et leur type.")

with st.expander(label="Afficher le tableau des variables", expanded=False):
    st.markdown("### Tableau des variables")
    st.write(df_table_description)

# ------------------------------------------------------------------------------------------------
# Gestion des valeurs manquantes
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("L'exploration des données nous a permis d'identifer que nous n'avons pas de valeur manquante. Par contre nous possèdons des valeurs qui ne nous parraisent pas exploitable en l'état")


with st.expander(label="Afficher le tableau des valeurs non désirées", expanded=False):
    st.markdown("### Tableau des valeurs non désirés")
    st.write(df_tableau_des_valeurs_non_désirées)


st.markdown(
    """
    Nous allons donc procéder à la suppression des <span class="orange-bold">**«unknown»**</span> des modalités des varibles <span class="orange-bold">job</span> et <span class="orange-bold">education</span>, car elles représentent un volume minime de notre jeu de données (respectivement 1% et 4%).

    Les variables <span class="orange-bold">pdays</span> et <span class="orange-bold">previous</span> decrivent la même chose, nous décidons donc de garder qu’une seule variable <span class="orange-bold">pdays</span>. Cette dernière nous apporte une information en plus, le nombre de jours écoulés depuis le dernier contact.

    Pour la variable <span class="orange-bold">poutcome</span>, nous décidons de regrouper les 2 modalités <span class="orange-bold">unknown</span> et <span class="orange-bold">other</span> sous une même modalité commune (<span class="orange-bold">unknown</span>), car il se peut que cela nous apporte une information supplémentaire lors de nos futures exploitations.

    La variable <span class="orange-bold">contact</span> qui a un grand nombre d’inconnues n’a aucun enjeux métier, elle est donc supprimée.
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003📊 Visualiation - Statistique"):
    st.switch_page("pages/3_📊_Visualisation_-_Statistique.py")
    
    
    
# ------------------------------------------------------------------------------------------------



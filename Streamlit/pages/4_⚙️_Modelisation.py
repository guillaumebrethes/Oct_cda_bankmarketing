import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
import matplotlib.pyplot as plt
from streamlit_shap import st_shap # type: ignore
import shap # type: ignore
import plotly.graph_objects as go # type: ignore

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="⚙️" 
)

st.title("Modélisation")

if st.button("◀️\u2003📊 Visualiation - Statistique"):
    st.switch_page("pages/3_📊_Visualiation_-_Statistique.py")
st.write("---")

st.markdown(
    """ 
    Introduction à ecrire 
    """
    )

# --------------------------------------------------------------------------------------------
# Importation des jeux d'entrainement et de test sauvegardés  depuis google collab
X_test = pd.read_csv("Split_csv/3_bank_X_test.csv",index_col=0)
y_test = pd.read_csv("Split_csv/3_bank_y_test.csv",index_col=0)
X_test_copie = pd.read_csv("Split_csv/3_bank_X_test_copie.csv",index_col=0)

#Importation des valeurs Shap (que la classe 1 pour rfc)
shap_values_gbc = np.load('Shap/shap_values_gbc.npy')
shap_values_rfc = np.load('Shap/shap_values_rfc.npy')


# importations des modèles optimisés à interpréter
gbc_after = joblib.load("Models/model_gbc_after")
rfc_after = joblib.load("Models/model_rfc_after")
# --------------------------------------------------------------------------------------------

st.write("---")
st.write("### Interprétation des Modèles avec la méthode SHAP ###")
# st.markdown('<h1 style="font-size: 30px;">Interprétation des Modèles avec la méthode SHAP</h1>', unsafe_allow_html=True)

#---------------------------------------
#Création d'un expander pour expliquer la méthode Shap

with st.expander("Cliquez ici pour en savoir plus sur la méthode SHAP"):
    st.markdown("""
    <div>
        La méthode SHAP (SHapley Additive exPlanations) repose sur les valeurs de Shapley, une méthode issue de la théorie des jeux coopératifs, pour attribuer à chaque caractéristique (ou variable) une importance en fonction de sa contribution à la prédiction.
        <br><br>
        SHAP est une méthode qui explique comment les prédictions individuelles sont effectuées par un modèle d'apprentissage automatique. Elle déconstruit une prédiction en une somme de contributions (valeurs SHAP) de chacune des variables d'entrée du modèle.
        <br><br>
        À noter que SHAP indique ce que fait le modèle dans le contexte des données sur lesquelles il a été formé. Il ne révèle pas nécessairement la véritable relation entre les variables et les résultats dans le monde réel.
    </div>
    """, unsafe_allow_html=True)
st.write("---")

#---------------------------------------
#On propose de voir la page en fonction du modèle séléctionné gbc_after ou rfc_after

# Sélection du modèle via liste déroulante
model_choice = st.selectbox(
    '**Sélectionner un modèle**',
    ['Gradiant Boosting Classifier', 'Random Forest Classifier'])

# Chargement des valeurs SHAP et explainer en fonction du modèle sélectionné
model = gbc_after if model_choice == 'Gradiant Boosting Classifier' else rfc_after

shap_values = shap_values_gbc if model_choice == 'Gradiant Boosting Classifier' else shap_values_rfc

expected_value = shap.TreeExplainer(gbc_after).expected_value \
    if model_choice == 'Gradiant Boosting Classifier'\
        else shap.TreeExplainer(rfc_after).expected_value[1]

#---------------------------------------
import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
import matplotlib.pyplot as plt
# from streamlit_shap import st_shap # type: ignore
# import shap # type: ignore
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

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
    page_icon="⚙️", 
    #layout="wide" 
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

st.write(shap_values_rfc.shape)

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

#----------------------------------------------------------------------------------------------------------------------
#On propose de voir la page en fonction du modèle séléctionné gbc_after ou rfc_after

# Sélection du modèle via liste déroulante
model_choice = st.selectbox(
    label='**Sélectionner un modèle pour afficher les 10 plus importantes variables**',
    options=['Gradiant Boosting Classifier', 'Random Forest Classifier'], 
    index = None, 
    placeholder= "Modèle . . .", 
    help= "Le chargement peut prendre entre 20 et 30 secondes")

# Chargement des valeurs SHAP et explainer en fonction du modèle sélectionné
model = gbc_after if model_choice == 'Gradiant Boosting Classifier' else rfc_after

shap_values = shap_values_gbc if model_choice == 'Gradiant Boosting Classifier' else shap_values_rfc

expected_value = shap.TreeExplainer(gbc_after).expected_value

if model_choice == 'Gradiant Boosting Classifier':
    explainer = shap.TreeExplainer(gbc_after)
    expected_value = explainer.expected_value
else:
    explainer = shap.TreeExplainer(rfc_after)
    expected_value = explainer.expected_value[1]

shap_values = explainer.shap_values(X_test)
plt.figure() 

shap.summary_plot(shap_values, 
                  X_test, 
                  plot_type="bar", 
                  max_display=10, 
                  show=False)

st.pyplot(plt.gcf(), use_container_width=True)
st.markdown(
    """
    L'axe des X représente la moyenne des valeurs SHAP absolues pour chaque variable, indiquant l'importance moyenne de chaque variable sur la prédiction du modèle. **duration est la variable qui influence le plus la prédiction du modèle (moyenne de 1.2)**
    """
    )

st.write("---")

#--------------------------------------------------------------------------------------------------------------

# Utilisation d'un extender pour montrer le Graphique d'importance des variables     
with st.expander("🔍 **Impact des variables dans la décision du modèle**"):
    plt.figure() 
    shap.summary_plot(shap_values, 
                      X_test, 
                      max_display=10, show=False)
    
    st.pyplot(plt.gcf(), use_container_width=True)
    st.markdown("""
Dans ce graphique, l'axe des x représente la valeur SHAP et l'axe des y représente les variables explicatives (ici le TOP 10). 

Chaque point du graphique correspond à une valeur SHAP pour une prédiction et une variable explicative. La couleur rouge signifie une valeur plus élevée de la variable explicative. Le bleu signifie une valeur faible de cette dernière. Nous pouvons avoir une idée générale de la directionnalité de l'impact des variables en fonction de la distribution des points rouges et bleus.

On peut lire que plus la valeur de **duration** est grande (le temps de l'appel long), plus l'impact sur la prédiction de souscription du dépôt à terme est positif  et inversement plus **duration** est faible, plus l'impact sur la prédiction est négatif. Une valeur importante de **poutcome_success** (client avait souscrit à un dépôt à terme auparavant) a un impact positif sur la souscription du dépôt à terme. 

Une valeur plus grande de **housing** (le client a un prêt immobilier) a un impact négatif sur la prédiction de la souscription du dépôt et inversement une valeur faible ( le client n’a pas de prêt immobilier) a un effet positif sur la prédiction de la souscription du dépôt.
""")

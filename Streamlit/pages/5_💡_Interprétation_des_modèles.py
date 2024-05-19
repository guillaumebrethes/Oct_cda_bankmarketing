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
    page_icon="💡", 
    #layout="wide" 
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

# titre
st.markdown('<h1 class="custom-title">Interprétation des modèles</h1>', unsafe_allow_html=True)

if st.button("◀️\u2003⚙️ Modelisation"):
    st.switch_page("pages/4_⚙️_Modelisation.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown(
    """ 
    Introduction à ecrire  Estelle
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

st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("### Interprétation des Modèles avec la méthode SHAP ###")
# st.markdown('<h1 style="font-size: 30px;">Interprétation des Modèles avec la méthode SHAP</h1>', unsafe_allow_html=True)

#---------------------------------------
#Création d'un expander pour expliquer la méthode Shap

with st.expander("Cliquez ici pour en savoir plus sur la méthode SHAP"):

    st.markdown("""
    <div class="explain_shap">
        La méthode SHAP (SHapley Additive exPlanations) repose sur les valeurs de Shapley, une méthode issue de la théorie des jeux coopératifs, pour attribuer à chaque caractéristique (ou variable) une importance en fonction de sa contribution à la prédiction.
        <br><br>
        SHAP est une méthode qui explique comment les prédictions individuelles sont effectuées par un modèle d'apprentissage automatique. Elle déconstruit une prédiction en une somme de contributions (valeurs SHAP) de chacune des variables d'entrée du modèle.
        <br><br>
        À noter que SHAP indique ce que fait le modèle dans le contexte des données sur lesquelles il a été formé. Il ne révèle pas nécessairement la véritable relation entre les variables et les résultats dans le monde réel.
    </div>
    """, unsafe_allow_html=True)
    
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------------------
#On propose de voir la page en fonction du modèle séléctionné gbc_after ou rfc_after

st.markdown("""
Bienvenue dans cette application d'analyse des modèles ! Vous pouvez sélectionner un modèle dans la liste déroulante ci-dessous pour visualiser les 10 variables les plus importantes.
""")

# Sélection du modèle via liste déroulante
model_choice = st.selectbox(
    label='',
    options=['Gradiant Boosting Classifier', 'Random Forest Classifier'], 
    index=None, 
    placeholder="Modèle . . .")
# ------------------------------------------

# Graphique d'importance des variables
if model_choice:
    model = gbc_after if model_choice == 'Gradiant Boosting Classifier' else rfc_after
    shap_values = shap_values_gbc if model_choice == 'Gradiant Boosting Classifier' else shap_values_rfc
    expected_value = shap.TreeExplainer(gbc_after).expected_value \
    if model_choice == 'Gradiant Boosting Classifier'\
        else shap.TreeExplainer(rfc_after).expected_value[1]

    plt.figure() 
    shap.summary_plot(shap_values, 
                      X_test, 
                      plot_type="bar", 
                      max_display=10, 
                      show=False)
    st.pyplot(plt.gcf(), use_container_width=True)

    st.markdown(
        """
        L'axe des X représente la moyenne des valeurs SHAP absolues pour chaque variable, indiquant l'importance moyenne de chaque variable sur la prédiction du modèle. **duration est la variable qui influence le plus la prédiction du modèle**
        """)

# Utilisation d'un extender pour montrer le Graphique d'importance des variables     
    with st.expander("🔍 **Impact des variables dans la décision du modèle**"):
        plt.figure() 
        shap.summary_plot(shap_values, 
                          X_test, 
                          max_display=10, 
                          show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
         
        st.markdown(
            """
            Dans ce graphique, l'axe des x représente la valeur SHAP et l'axe des y représente les variables explicatives (ici le TOP 10). Chaque point du graphique correspond à une valeur SHAP pour une prédiction et une variable explicative. 
            
            La couleur rouge signifie une valeur plus élevée de la variable explicative. Le bleu signifie une valeur faible de cette dernière. Nous pouvons avoir une idée générale de la directionnalité de l'impact des variables en fonction de la distribution des points rouges et bleus. 
            
            
            On peut lire que plus la valeur de **duration** est grande (le temps de l'appel long), plus l'impact sur la prédiction de souscription du dépôt à terme est positif  et inversement plus **duration** est faible, plus l'impact sur la prédiction est négatif.
            
            Une valeur importante de **poutcome_success** (client avait souscrit à un dépôt à terme auparavant) a un impact positif sur la souscription du dépôt à terme.
            
            Une valeur plus grande de de **housing** (le client a un prêt immobilier) a un impact négatif sur la prédiction de la souscription du dépôt et inversement une valeur faible ( le client n’a pas de prêt immobilier) a un effet positif sur la prédiction de la souscription du dépôt.
            """
            )

#---------------------------------------
# Utilisation d'un extender pour montrer les predictions et shap values par individu
    with st.expander("🔍 **Visualisation des prédictions individuelles du jeux de données Test**"):
    
    # Choix de l'index par l'utilisateur 743 est cool
        index_to_show = st.slider('Choisissez l\'index de l\'observation à visualiser', 0, len(X_test) - 1, 0, 
                              help = "les cases cochées représentent le mois, le job, le poutcome du client selectionné")

    # Créer un objet Explanationx
        shap_values_instance = shap.Explanation(
            values=shap_values[index_to_show],
            base_values=expected_value,
            data=X_test.iloc[index_to_show]  # Inclure les données d'entrée pour plus de contexte dans le plot
    )

    #Afficher les informations du DataFrame pour cet infividu
        st.dataframe(X_test_copie.iloc[[index_to_show]])

    # Afficher (y_test) ---
        if y_test.iloc[index_to_show].item()== 1:
            deposit = "**a souscrit au dépôt à terme**"
        else:
            deposit = "**n'a pas souscrit au dépôt à terme**"
        st.markdown(f"**Décision réelle du client** : Cet individu {deposit}")
    # ---
    
    # Afficher ((y_pred) ---
        y_pred = model.predict(X_test.iloc[[index_to_show]])
        if y_pred.item() == 1:
           deposit_pred = " **souscrira au dépôt à terme**"
        else:
            deposit_pred = "**ne souscrira pas au dépôt à terme**"
        st.markdown(f"**Prédiction du modèle** : Ce modèle prédit que cet individu {deposit_pred}")
    # ---

    #Afficher les valeurs shap (top 10 pour cet individu)
        st.markdown(' **Waterfall plot** pour cet individu : ')
    
    # Create a figure 
        fig, ax = plt.subplots()

    # Generate the waterfall plot on the created figure
        shap.plots.waterfall(shap_values_instance, max_display=10, show=False)  

    # Display the plot in Streamlit
        st.pyplot(fig)

   #Explications de lecture du graphique
   
        st.markdown(
        """
        La structure en cascade illustre comment les contributions additives des variables explicatives, qu'elles soient positives ou négatives, s'accumulent à partir d'une valeur de base (E[f(X)]). 
        
        Cette accumulation met en évidence comment chaque variable explicative construit progressivement la prédiction finale du modèle, notée f(x).
        """
        ) 

#---------------------------------------

    with st.expander("🔍 **Impact des variables dans la prédiction en fonction de leur valeur**"):
    
        st.markdown("Grace au **Dependance plot** on peut visualiser et comprendre comment des valeurs spécifiques d'une variable influencent les prédictions du modèle.")
   
    # Case à cocher pour le  graphique
        if st.checkbox("Dependance Plot **duration**", key='checkbox5'):
            plt.figure()
            shap.dependence_plot('duration', shap_values, X_test_copie, interaction_index=None)
            st.pyplot(plt)
            st.markdown("• On peut voir à partir de quelle valeur de **duration** l'impact sur la prédiction devient positif")
        
    # Case à cocher pour le  graphique
        if st.checkbox("Dependance Plot **balance**", key='checkbox6'):
            st.markdown("**balance** est le solde moyen annuel sur le compte courant")
            plt.figure()
            shap.dependence_plot('balance', shap_values, X_test_copie, interaction_index=None)
            st.pyplot(plt)
            st.markdown("• On peut voir à partir de quelle valeur de **balance** l'impact sur la prédiction devient positif")
        
    # Case à cocher pour le  graphique
        if st.checkbox("Dependance Plot **age**", key='checkbox7'):
            plt.figure()
            shap.dependence_plot('age', shap_values, X_test_copie, interaction_index=None)
            st.pyplot(plt)
            st.markdown("• On peut voir les **ages** pour lesquels l'impact sur la prédiction est positif et ceux pour lesquels l'impact est négatif")
        
    # Case à cocher pour le  graphique
        if st.checkbox("Dependance Plot **campaign**", key='checkbox8'):
            st.markdown("**campaign** est le nombre de contacts effectués sur la campagne")
            plt.figure()
            shap.dependence_plot('campaign', shap_values, X_test_copie, interaction_index=None)
            st.pyplot(plt)        
            st.markdown("• On peut voir que s'il y a plus d'1 contact, **campaign** a un impact négatif sur la prévision")
            


# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003 🎯 Recommandation métier - Conclusion"):
    st.switch_page("pages/6_🎯_Recommandation_métier_-_Conclusion.py")
    

# ------------------------------------------------------------------------------------------------

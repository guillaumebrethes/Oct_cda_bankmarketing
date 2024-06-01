import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
import matplotlib.pyplot as plt
from streamlit_shap import st_shap # type: ignore
import shap # type: ignore
import plotly.graph_objects as go # type: ignore
import matplotlib.colors as mcolors

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

lien_methodeShap = "https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html"


st.markdown(
    f""" Pour compléter l'analyse de la performance des modèles, nous allons interpréter la façon dont les modèles prédisent en utilisant la méthode <a href="{lien_methodeShap}" class="orange-bold">SHAP (SHapley Additive exPlanations)</a>. """, unsafe_allow_html=True
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
st.markdown("<h3 class='titre-h3'>Interprétation des Modèles avec la méthode SHAP</h3>", unsafe_allow_html=True)

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
    options=['Gradient Boosting Classifier', 'Random Forest Classifier'], 
    index=None, 
    placeholder="Modèle . . .")
# ------------------------------------------

# Graphique d'importance des variables
if model_choice:
    model = gbc_after if model_choice == 'Gradient Boosting Classifier' else rfc_after
    shap_values = shap_values_gbc if model_choice == 'Gradient Boosting Classifier' else shap_values_rfc
    expected_value = shap.TreeExplainer(gbc_after).expected_value \
    if model_choice == 'Gradient Boosting Classifier'\
        else shap.TreeExplainer(rfc_after).expected_value[1]

   # Créez la figure avec un fond transparent
    fig = plt.figure()
    fig.patch.set_alpha(0)  # Rendre le fond de la figure transparent
   
   # Générer le graphique SHAP de type "bar"
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10, show=False, color='#ADD8E6')
   
   # Rendre le fond des axes transparent
    ax = plt.gca()
    ax.patch.set_alpha(0)
   
   # Afficher le graphique avec Streamlit en utilisant un fond transparent
    plt.gcf().set_facecolor('none')
    st.pyplot(plt.gcf(), use_container_width=True)

    st.markdown(
        """
        L'axe des X représente la moyenne des valeurs SHAP absolues pour chaque variable, indiquant l'importance moyenne de chaque variable sur la prédiction du modèle. **`duration`**  est la variable qui influence le plus la prédiction du modèle.
        """, unsafe_allow_html=True)

# Utilisation d'un extender pour montrer le Graphique d'importance des variables     
    with st.expander("🔍 **Impact des variables dans la décision du modèle**"):
        fig = plt.figure()
        fig.patch.set_alpha(0)  # Rendre le fond de la figure transparent
        
        # Définir la colormap personnalisée
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["#ADD8E6", "#F08080"])
      
        # Générer le graphique SHAP
        shap.summary_plot(shap_values, X_test, max_display=10, show=False, cmap=cmap)
        
        # Rendre le fond des axes transparent
        ax = plt.gca()
        ax.patch.set_alpha(0)
        
        # Afficher le graphique avec Streamlit en utilisant un fond transparent
        plt.gcf().set_facecolor('none')
        st.pyplot(plt.gcf(), use_container_width=True)
         
        st.markdown(
            """
            Dans ce graphique, l'axe des x représente la valeur SHAP et l'axe des y représente les variables explicatives (ici le TOP 10). Chaque point du graphique correspond à une valeur SHAP pour une prédiction et une variable explicative. 
            
            La couleur rouge signifie une valeur plus élevée de la variable explicative. Le bleu signifie une valeur faible de cette dernière. Nous pouvons avoir une idée générale de la directionnalité de l'impact des variables en fonction de la distribution des points rouges et bleus. 
            
            
            On peut lire que plus la valeur de **`duration`** est grande (temps de l'appel long), plus l'impact sur la prédiction de souscription du dépôt à terme est positif  et inversement plus **`duration`** est faible, plus l'impact sur la prédiction est négatif.
            
            Une valeur importante de **`poutcome_success`** (client avait souscrit à un dépôt à terme auparavant) a un impact positif sur la souscription du dépôt à terme.
            
            Une valeur plus grande de **`housing`** (le client a un prêt immobilier) a un impact négatif sur la prédiction de la souscription du dépôt et inversement une valeur faible ( le client n’a pas de prêt immobilier) a un effet positif sur la prédiction de la souscription du dépôt.
            """, unsafe_allow_html=True)

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
            deposit = "**a souscrit au dépôt à terme.**"
        else:
            deposit = "**n'a pas souscrit au dépôt à terme.**"
        st.markdown(f"<span class='orange-bold'>Décision réelle du client</span> : Cet individu <span style='font-weight:bold; color:black;'>{deposit}</span>", unsafe_allow_html=True)
    # ---
    
    # Afficher ((y_pred) ---
        y_pred = model.predict(X_test.iloc[[index_to_show]])
        if y_pred.item() == 1:
           deposit_pred = " **souscrira au dépôt à terme.**"
        else:
            deposit_pred = "**ne souscrira pas au dépôt à terme.**"
        st.markdown(f"<span class='orange-bold'>Prédiction du modèle</span> : Ce modèle prédit que cet individu <span style='font-weight:bold; color:black;'>{deposit_pred}</span>", unsafe_allow_html=True)
    # ---

    #Afficher les valeurs shap (top 10 pour cet individu)
        st.markdown("<span class='orange-bold'>Waterfall plot</span> pour cet individu :", unsafe_allow_html=True)
        # Create a figure with transparent background
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # Generate the waterfall plot on the created figure
        shap.plots.waterfall(shap_values_instance, max_display=10, show=False)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
   #Explications de lecture du graphique
   
           #Explications de lecture du graphique
           
        st.markdown(
        """
        La structure en cascade illustre comment les **`contributions additives des variables explicatives`** , qu'elles soient positives ou négatives, s'accumulent à partir d'une valeur de base (E[f(X)]). 
        
        Cette accumulation met en évidence comment chaque variable explicative construit progressivement la prédiction finale du modèle, notée f(x).
        """
        ) 

#---------------------------------------

    with st.expander("🔍 **Impact des variables dans la prédiction en fonction de leur valeur**"):
    
        st.markdown("""Grâce au <span class="orange-bold">Dependance plot</span> on peut visualiser et comprendre comment des valeurs spécifiques d'une variable influencent les prédictions du modèle.""", unsafe_allow_html=True)
            
        # Fonction pour afficher le graphique de dépendance SHAP
        def plot_dependence_plot(selected_var, shap_values, X_test_copie, fig_width=4, fig_height=3):
            plt.figure(figsize=(fig_width, fig_height), facecolor='none')
            shap.dependence_plot(selected_var, shap_values, X_test_copie, interaction_index=None, color='#ADD8E6', show=False)
            ax = plt.gca() 
            ax.patch.set_alpha(0)
            plt.gcf().set_facecolor('none')
            st.pyplot(plt)

        # Liste des variables à afficher dans le multiselect
        variables = ['duration', 'balance', 'age', 'campaign']

        # Affichage du multiselect
        selected_variables = st.multiselect("Sélectionnez les variables à afficher :", variables)

        # Affichage des graphiques pour les variables sélectionnées
        for var in selected_variables:
            st.markdown(f"""<span style="font-size: 20px; font-weight: bold; color: #E97132; text-decoration: underline;">Graphique de Dépendance SHAP pour la Variable **`{var}`**</span>""", unsafe_allow_html=True)
            if var=='balance':
                st.markdown("""<span class="orange-bold">balance</span> est le solde moyen annuel sur le compte courant.""", unsafe_allow_html=True)
            elif var=='campaign':
                st.markdown("""<span class="orange-bold">campaign</span> est le nombre de contacts effectués sur la campagne.""", unsafe_allow_html=True)
            
            plot_dependence_plot(var, shap_values, X_test_copie, fig_width=3, fig_height=2)
            
            if var == 'duration':
                st.markdown("""On peut voir à partir de quelle valeur de **`duration`** l'impact sur la prédiction devient positif.""", unsafe_allow_html=True)
            elif var=='balance':
                st.markdown("""On peut voir à partir de quelle valeur de **`balance`** l'impact sur la prédiction devient positif.""", unsafe_allow_html=True)
            elif var=='age':
                st.markdown("""On peut voir les **`âges`** pour lesquels l'impact sur la prédiction est positif et ceux pour lesquels l'impact est négatif.""", unsafe_allow_html=True)
            elif var=='campaign':
                st.markdown("""On peut voir que s'il y a plus d'1 contact, **`campaign`** a un impact négatif sur la prévision.""", unsafe_allow_html=True)
            st.markdown("---")    

# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003 🎯 Recommandation métier - Conclusion"):
    st.switch_page("pages/6_🎯_Recommandation_métier_-_Conclusion.py")
    

# ------------------------------------------------------------------------------------------------

import streamlit as st  # type: ignore
from streamlit_shap import st_shap # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import shap # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.metrics import roc_curve, roc_auc_score # type: ignore

# Variables 
df_tableau_diff_analyse = pd.read_csv("Tableau_des_différentes_analyses.csv", sep=";")

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="⚙️", 
    #layout="wide" 
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

st.markdown('<h1 class="custom-title">Modélisation</h1>', unsafe_allow_html=True)
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


if st.button("◀️\u2003📊 Visualiation - Statistique"):
    st.switch_page("pages/3_📊_Visualiation_-_Statistique.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown(
    """ 
    Introduction à ecrire 
    """
    )

# --------------------------------------------------------------------------------------------

df=pd.read_csv("bank.csv")
dfclean=pd.read_csv("2_bank_clean.csv")
# Importation des jeux d'entrainement et de test sauvegardés  depuis google collab
X_test = pd.read_csv("Split_csv/3_bank_X_test.csv",index_col=0)
y_test = pd.read_csv("Split_csv/3_bank_y_test.csv",index_col=0)
X_train= pd.read_csv("Split_csv/3_bank_X_train.csv",index_col=0)
y_train= pd.read_csv("Split_csv/3_bank_y_train.csv",index_col=0)
X_test_copie = pd.read_csv("Split_csv/3_bank_X_test_copie.csv",index_col=0)

#Importation des valeurs Shap (que la classe 1 pour rfc)
shap_values_gbc = np.load('Shap/shap_values_gbc.npy')
shap_values_rfc = np.load('Shap/shap_values_rfc.npy')

# importations des modèles optimisés à interpréter
gbc_after = joblib.load("Models/model_gbc_after")
rfc_after = joblib.load("Models/model_rfc_after")

# importations des modèlesavant optimisation
gbc_before = joblib.load("Models/model_gbc_before")
rfc_before = joblib.load("Models/model_rfc_before")
# --------------------------------------------------------------------------------------------

# PRE PROCESSING

st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Pré Processing</h3>", unsafe_allow_html=True)


with st.expander("Cliquez ici pour en savoir plus sur la Transformation du Data Frame pour l'étape de machine Learning"):
    
# Gestion des Outliers
    if st.checkbox("Gestion des Valeur Extrêmes", key='checkbox1'):
        st.markdown("Il n'y a aucune valeurs extrêmes qui semblent abérentes dans nos variables qualitatives. Cependant nous devons traiter les valeurs extrêmes pour éviter les perturbations sur nos modèles de Machine Learning.")
        st.markdown(
            """ 
            **Nous appliquons la méthode <span class="orange-bold">"IQR"</span> :**  
            on supprime les valeurs qui se trouvent en dehors de l'intervalle "Inter Quartile Range", c'est à dire :
            - les valeurs supérieures à [ Q3 + 1.5 x (Q3 - Q1)]
            - les valeurs inférieures à [Q1 - 1.5 x (Q3 - Q1)]  
            avec Q1 le premier quartile et Q3 le troisième quartile
            """, unsafe_allow_html=True)
        
        st.write("Nous avons supprimé", round((100 - (dfclean.shape[0] * 100) / df.shape[0]), 2), "*%* des lignes de notre dataframe initial", "cependant il nous reste encore :", dfclean.shape[0], "lignes (clients) pour le reste des études")


    # Définition de la fonction plot_box pour afficher les boxplots d'une seule colonne
        def plot_box(df, 
                     column, 
                     fig_width=600, 
                     fig_height=300, 
                     color='skyblue', 
                     title_suffix=""):
            fig = px.box(df, 
                         x=column, 
                         hover_data=df.columns)
            fig.update_layout(title=f"<b>Boxplot de '{column}' {title_suffix}</b>", width=fig_width, height=fig_height)
            fig.update_traces(marker=dict(color=color))
            st.plotly_chart(fig)

   # Liste des variables pour lesquelles on veut voir les boxplots avec une option initiale
        variables = ['age', 'balance', 'duration', 'campaign']
        selected_var = st.selectbox(
            "Choisir une variable pour afficher les boîtes à moustache avant et après suppression des valeurs extrêmes:", 
            options = variables,
            index = None,
            placeholder = "Variables . . .")

    # Vérification si une variable a été sélectionnée et n'est pas l'option initiale
        if selected_var :
            plot_box(df, 
                     selected_var, 
                     fig_width=600, 
                     fig_height=300, 
                     color='lightcoral', 
                     title_suffix="avant supression des valeurs extrêmes")
    
            plot_box(dfclean, 
                     selected_var, 
                     fig_width=600, 
                     fig_height=300, 
                     color='lightcoral', 
                     title_suffix="après suppression des valeurs extrêmes")

# Encodage des variables
    if st.checkbox("Encodage des variables", key='checkbox2'):
   
        # Variables Binaires 
        st.markdown("<strong class='type-de-variables'>🗂️ Variables Binaires</strong>", unsafe_allow_html=True)
        st.markdown(
            """
            - Les modalités **`yes`** et **`no`** des variables **`default`**, **`housing`**, **`loan`**, **`deposit`** seront donc remplacées respectivement par **`1`** et **`0`**
            - Nous avons arbitrairement remplacé la modalité **`-1`** de **`pdays`**  par **`0`**, pour faciliter la compréhension d'un point de vue métier. En effet, si il n'y a pas eu de contact depuis la précédente campagne marketing, la valeur la plus adaptée semble être **`0`**
            """
            )
        
        # Variables ordinales 
        st.markdown("<strong class='type-de-variables'>🗂️ Variables ordinales</strong>", unsafe_allow_html=True)

        st.markdown("- La seule variable ordinale dans le jeu de données est **`education`**. Nous décidons de remplacer les modalités : **`primary`**, **`secondary`** et **`tertiary`**, respectivement par **`0`**, **`1`** et **`2`**.")
        
        # Variables non-ordinales
        st.markdown("<strong class='type-de-variables'>🗂️ Variables non-ordinales</strong>", unsafe_allow_html=True)
        st.markdown(
            """
            - Pour les variables **`job`**, **`marital`**, **`month`**, **`poutcome`** qui sont non-ordinales, nous allons appliquer la méthode **`get.dummies()`**, pour effectuer une dichotomisation.
            - Avant cela, nous avons bien évidemment séparé notre variable cible **`y (deposit)`** de notre jeu de données **`X`**. Nous avons réalisé un split entre le jeu d'entraînement **` X_TRAIN (80%)`** et le jeu de test **`X_TEST (20%)`**. 
            """
            )

        st.write("➡️ la taille de notre df initial est de :",df.shape)
        st.write("➡️ la taille de notre df X_train est de :", X_train.shape)
        st.write("➡️ la taille de notre df X_test est de :", X_test.shape)
    

# Standardisation des données   
    if st.checkbox("Standardisation des données", key='checkbox3'):
        lien_standartScaller = "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"
    
        st.markdown(
            """
            Nous utilisons <a href="{lien_standartScaller}" class="orange-bold">StandartScaler()</a>, qui nous permet de réaliser une mise à l'échelle en soustrayant la moyenne et en divisant par l'écart type, de sorte que les valeurs aient une moyenne de zéro, et un écart type de 1.
            """, unsafe_allow_html=True)

        

#--------------------------------------------------------------------------------------------
  
  
#MODELISATION

st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Sélection et Optimisation des Modèles</h3>", unsafe_allow_html=True)

# st.markdown('<h1 style="font-size: 30px;">Interprétation des Modèles avec la méthode SHAP</h1>', unsafe_allow_html=True)

#Synthèse des étapes de modélisation et présentation du tableau de résultats
with st.expander("Cliquez ici pour en savoir plus sur les étapes de la modélisation"):

    st.markdown(
        """<strong class='type-de-variables'>📝 Problématique</strong> 
                
Ce projet s'apparente à une tâche de machine learning appelée **`la classification supervisée`**. La classification consiste à prédire si un client (**la variable à prédire**) acceptera (**classe 1**) ou non (**classe 0**) de souscrire à un dépôt bancaire en utilisant les données disponibles sur ce client.
""", unsafe_allow_html=True)

    st.markdown(
        """ <strong class='type-de-variables'>📏 Métrique</strong>

Nous choisissons le **`Recall de la classe 1`** comme métrique clé dans **l'évaluation** de nos modèles.  
↗️ Maximiser les **Vrais positifs** (identifications correctes de clients potentiels qui sont très susceptibles de souscrire à l'offre)  
↘️ Minimiser les **Faux Négatifs** (le nombre de ces clients potentiels que le modèle pourrait manquer)
""", unsafe_allow_html=True)

    st.markdown(
        """ <strong class='type-de-variables'>⚙️ Méthode d'optimisation des hyperparamètres</strong>
          
Nous utilisons **`GridSearchCV()`** pour trouver la combinaison optimale des paramètres des modèles.
""", unsafe_allow_html=True)

    st.markdown("""
✔️ **Modèles entrainés et optimisés**  
  1️⃣ Random Forest Classifier<br>
  2️⃣ Gradiant Boosting Classifier<br>
  3️⃣ Decision Tree Classifier<br>
  4️⃣ SVM Classifier<br>
  5️⃣ Regression<br>

""", unsafe_allow_html=True)

#On présente le tableau des résultats avec un bouton qui s'ouvre ou se ferme
    # Initialisation de la variable d'état si elle n'existe pas déjà
    if 'show_image' not in st.session_state:
        st.session_state.show_image = False

    # Définition du bouton
    if st.button("🎯 Tableau de résultats de la modélisation", key='button4'):
        # Toggle de l'état
        st.session_state.show_image = not st.session_state.show_image

    # Condition pour afficher ou non l'image
    if st.session_state.show_image:
    # Afficher les lignes sélectionnées du DataFrame
        st.write(df_tableau_diff_analyse)

# ----------------------------------------------------------------------
#ANALYSE PAPPROFONDIE DES TOPS MODELES

st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Analyse Approfondie des Top Modèles</h3>", unsafe_allow_html=True)

# st.markdown('<h1 style="font-size: 30px;">Interprétation des Modèles avec la méthode SHAP</h1>', unsafe_allow_html=True)

#On propose de voir la page en fonction du modèle séléctionné gbc_after ou rfc_after

st.markdown("""
Bienvenue dans cette application d'analyse des modèles ! Vous pouvez sélectionner un modèle dans la liste déroulante ci-dessous pour découvrir ce modèle en détails.
""")

# Sélection du modèle via liste déroulante
model_choice = st.selectbox(
    label='Sélectionner un modèle',
    options=['Gradiant Boosting Classifier', 'Random Forest Classifier'], 
    index=None,  # Assurez-vous également que l'index est valide, 0 pour sélectionner le premier élément
    placeholder="Modèle . . .")  # Masquer le label tout en restant accessible
# ------------------------------------------

if model_choice:
    model_after = gbc_after if model_choice == 'Gradiant Boosting Classifier' else rfc_after
    model_before = gbc_before if model_choice == 'Gradiant Boosting Classifier' else rfc_before
        
    #Présentation du modèle
    st.markdown("...................under construction..................;;")
            
    #Performance du modèle       
    if st.checkbox("Performance du Modèle avant et après Optimisation", key='checkbox9'):
        st.markdown('under construction')
        def display_model_performance(model, title):
            st.header(title)
            
            # Affichage des scores
            train_score = "{:.4f}".format(model.score(X_train, y_train))
            test_score = "{:.4f}".format(model.score(X_test, y_test))
            st.write(f"Score sur ensemble train: {train_score}")
            st.write(f"Score sur ensemble test: {test_score}")
        
            # Prédiction et rapport de classification
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            st.code(f"Rapport de classification :\n{report}")
        
            # Calcul de la matrice de confusion
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(conf_matrix, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues', 
                        cbar=False, 
                        ax=ax)
            ax.set_title('Heatmap de la Matrice de Confusion')
            ax.set_xlabel('Prédictions')
            ax.set_ylabel('Véritables Classes')
            st.pyplot(fig)
        
            # Création de deux colonnes pour les modèles
            col1, col2 = st.columns(2)
        
        
            # Affichage du modèle avant dans la première colonne
            with col1:
                display_model_performance(model_before, "Modèle Avant")
        
            # Affichage du modèle après dans la deuxième colonne
            with col2:
                display_model_performance(model_after, "Modèle Après")
        
        
        
        #Courbe ROC
            # Prédire les scores de probabilité
            y_scores_before = model_before.predict_proba(X_test)[:, 1]  # Score pour la classe positive
            y_scores_after = model_after.predict_proba(X_test)[:, 1]
        
            # Calcul des courbes ROC
            fpr_before, tpr_before, _ = roc_curve(y_test, y_scores_before)
            fpr_after, tpr_after, _ = roc_curve(y_test, y_scores_after)
        
            # Tracer les courbes ROC
            fig, ax = plt.subplots()
            ax.plot(fpr_before, 
                    tpr_before, 
                    label=f'ROC Modèle Avant (AUC = {roc_auc_score(y_test, y_scores_before):.2f})')
            ax.plot(fpr_after,
                    tpr_after, 
                    label=f'ROC Modèle Après (AUC = {roc_auc_score(y_test, y_scores_after):.2f})')
            ax.set_title('Comparaison des Courbes ROC')
            ax.set_xlabel('Taux de Faux Positifs')
            ax.set_ylabel('Taux de Vrais Positifs')
            ax.legend(loc='lower right')
            ax.grid(True)
        
            st.pyplot(fig)
    
# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
if st.button("▶️\u2003 💡 Interprétation_des_modèles"):
    st.switch_page("pages/5_💡_Interprétation_des_modèles.py")
    

# ------------------------------------------------------------------------------------------------

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
    Nous exposons ici notre travail de modélisation. Nous allons effecter le prétraitement des données, choisir une métrique de performance, entraîner et ajuster des modèles. Pour finir, nous présenterons l'analyse des modèles les plus performants.
    """
    )

# --------------------------------------------------------------------------------------------

df=pd.read_csv("bank.csv") #jeu de données initial
dfclean=pd.read_csv("2_bank_clean.csv")#jeu de données nettoyé et encodé

#résultats de tous les modèles entrainés
df_tableau_diff_analyse = pd.read_csv("Tableau_des_différentes_analyses.csv", sep=";")

#gbc et rfc paramètres before & after
params_gbc_before_df = pd.read_csv("Models/params_gbc_before.csv", sep=",")
params_gbc_after_df = pd.read_csv("Models/params_gbc_after.csv", sep=",")
params_rfc_before_df = pd.read_csv("Models/params_rfc_before.csv", sep=",")
params_rfc_after_df = pd.read_csv("Models/params_rfc_after.csv", sep=",")


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

# importations des modèles avant optimisation
gbc_before = joblib.load("Models/model_gbc_before")
rfc_before = joblib.load("Models/model_rfc_before")

# importations des grilles de paramètre utilisées dand GriSearchCV()
gbc_param_grid_df = pd.read_csv('Models/gbc_param_grid.csv')
rfc_param_grid_df = pd.read_csv('Models/rfc_param_grid.csv')

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
            **`Nous appliquons la méthode "IQR":`**  
            on supprime les valeurs qui se trouvent en dehors de l'intervalle "Inter Quartile Range", c'est à dire :
            - les valeurs supérieures à [ Q3 + 1.5 x (Q3 - Q1)]
            - les valeurs inférieures à [Q1 - 1.5 x (Q3 - Q1)]  
            avec Q1 le premier quartile et Q3 le troisième quartile
            """, unsafe_allow_html=True)
        
        st.write("Nous avons supprimé", round((100 - (dfclean.shape[0] * 100) / df.shape[0]), 2), "*%* des lignes de notre dataframe initial", "cependant il nous reste encore :", dfclean.shape[0], "lignes (clients) pour l'étape de modélisation.")
        
        def plot_box(df, column, fig_width=600, fig_height=300, color='skyblue', title_suffix=""):
            # Créer le graphique Boxplot avec Plotly
            fig = px.box(df, x=column, hover_data=df.columns)
        
            # Mise à jour de la mise en page
            fig.update_layout(
                title=f"<b>Boxplot de '{column}' {title_suffix}</b>",
                width=fig_width,
                height=fig_height,
                font_family="Arial",  # Harmonisation de la police
                title_font_family="Arial",
                paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                xaxis_title=f"<b style='color:black; font-size:90%;'>{column}</b>",
                yaxis_title="<b style='color:black; font-size:90%;'>Valeur</b>"
            )
        
           # Mise à jour des traces
            fig.update_traces(
                marker=dict(
                  color=color,  # Couleur du remplissage
                  line=dict(color='gray', width=1)  # Couleur de la bordure des points et son épaisseur
              ),
              boxpoints='outliers'  # Afficher uniquement les points aberrants (outliers)
          )
        
            # Affichage du graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

# Exemple d'utilisation de la fonction (remarque : assurez-vous que 'df' et 'column' sont définis dans votre script)
# plot_box(df, 'some_column_name', title_suffix="quelque chose")
        
        

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
            - Avant cela, nous avons bien évidemment séparé notre variable cible **`y (deposit)`** de notre jeu de données **`X`**. Nous avons réalisé un split entre le jeu d'entraînement **` X_TRAIN (80%)`** et le jeu test **`X_TEST (20%)`**. 
            """
            )

        st.write("➡️ la taille de notre df initial est de :",df.shape)
        st.write("➡️ la taille de notre df X_train est de :", X_train.shape)
        st.write("➡️ la taille de notre df X_test est de :", X_test.shape)
    

# Standardisation des données   

    if st.checkbox("Standardisation des données", key='checkbox3'):
        lien_standartScaller = "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"
        
        st.markdown(
            f"""
            Nous utilisons <a href="{lien_standartScaller}" class="orange-bold">StandardScaler()</a>, qui nous permet de réaliser une mise à l'échelle en soustrayant la moyenne et en divisant par l'écart type, de sorte que les valeurs aient une moyenne de zéro, et un écart type de 1.
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

Nous choisissons le **`Recall de la classe 1`** (Rappel) comme métrique clé dans **l'évaluation** de nos modèles.  
↗️ Maximiser les **Vrais positifs** (identifications correctes de clients potentiels qui sont très susceptibles de souscrire à l'offre)  
↘️ Minimiser les **Faux Négatifs** (le nombre de ces clients potentiels que le modèle pourrait manquer)
""", unsafe_allow_html=True)

    lien_GridSearchCV = "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"



    st.markdown(
        f""" <strong class='type-de-variables'>⚙️ Méthode d'optimisation des hyperparamètres</strong>
          
Nous utilisons <a href="{lien_GridSearchCV}"> GridSearchCV()</a> pour trouver la combinaison optimale des paramètres des modèles.
""", unsafe_allow_html=True)

    st.markdown("""<strong class='type-de-variables'>
✔️ Modèles entrainés et optimisés</strong><br>
  1️⃣ Random Forest Classifier<br>
  2️⃣ Gradiant Boosting Classifier<br>
  3️⃣ Decision Tree Classifier<br>
  4️⃣ SVM Classifier<br>
  5️⃣ Regression<br>

""", unsafe_allow_html=True)

    
        # Initialisation des variables d'état si elles n'existent pas déjà
    if 'show_table' not in st.session_state:
        st.session_state.show_table = False
    if 'show_visualization' not in st.session_state:
        st.session_state.show_visualization = False
  
#Bouton de visualisation des résultas
    # Définition des boutons
    if st.button("📊 Visualisation des résultats", key='button5'):
        st.session_state.show_visualization = not st.session_state.show_visualization
    
    # Condition pour afficher ou non la visualisation des résultats
    if st.session_state.show_visualization:
      
        # Filtrer les données par "par défaut" et "personnalisé"
        default = df_tableau_diff_analyse[df_tableau_diff_analyse['Hyper parametres'] == 'par défaut']
        custom = df_tableau_diff_analyse[df_tableau_diff_analyse['Hyper parametres'] == 'Optimisé']
        
        # Créer un DataFrame pour Plotly
        df_plotly = pd.concat([
            default.assign(HyperParam="Par défaut"),
            custom.assign(HyperParam="Optimisé")
        ])
        
        # Ajouter la colonne de pourcentage pour les étiquettes de texte
        df_plotly['Recall % class 1'] = df_plotly['Recall % class 1'] 
        df_plotly['percent'] = df_plotly['Recall % class 1'].apply(lambda x: '{}%'.format(round(x, 1)))
        
        # Créer le graphique à barres avec Plotly Express
        fig = px.bar(df_plotly, x='Modèles', y='Recall % class 1', text='percent', color='HyperParam',
                     barmode='group', color_discrete_sequence=['lightcoral', 'lightblue'],
                     width=600, height=450)
        
        # Mettre à jour les traces
        fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition="outside")
        
        # Mettre à jour la mise en page de la figure
        fig.update_layout(
            showlegend=True,
            title_text='<b style="color:black; font-size:90%;">Comparaison des modèles par défaut et Optimisés sur Recall % class 1</b>',
            font_family="Arial",
            title_font_family="Arial",
            xaxis_title='<b style="color:black; font-size:90%;">Modèles</b>',
            yaxis_title='<b style="color:black; font-size:90%;">Recall % class 1</b>',
            yaxis=dict(range=[0, 110]),  # Ajuster l'axe des ordonnées
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
 )
        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
  
#Bouton Détail des résultats(Tableau)
    if st.button("𝄜 Tableau Détaillé des résultats", key='button4'):
        st.session_state.show_table = not st.session_state.show_table
    
    # Condition pour afficher ou non le tableau des résultats
    if st.session_state.show_table:
        # Afficher les lignes sélectionnées du DataFrame
        cols_to_format = [
            "% Score Train",
            "% Score Test",
            "Recall % class 0",
            "Recall % class 1"
        ]
    
        # Appliquer le formatage
        df_tableau_diff_analyse_style = df_tableau_diff_analyse.style.format({col: "{:.2f}" for col in cols_to_format})
        
        # Afficher le DataFrame mis en forme
        st.write(df_tableau_diff_analyse_style)




# ----------------------------------------------------------------------
#ANALYSE APPROFONDIE DES TOPS MODELES

st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Analyse Approfondie des Top Modèles</h3>", unsafe_allow_html=True)

# st.markdown('<h1 style="font-size: 30px;">Interprétation des Modèles avec la méthode SHAP</h1>', unsafe_allow_html=True)

#On propose de voir la page en fonction du modèle séléctionné gbc_after ou rfc_after

st.markdown("""
Bienvenue dans cette application d'analyse des modèles ! Vous pouvez sélectionner un modèle dans la liste déroulante ci-dessous pour découvrir ses détails.
""")

#Fonction qui englobe tout le display en fonction du modèle choisi
def display_model_analysis(model, title, description=False, parameters=False, performance=False):
    # st.header(title)
    
    if description:
        if title == "Gradient Boosting Classifier":
            lien_GBC = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"
                   
            st.markdown(f"""
            <a href="{lien_GBC}">Le Gradient Boosting Classifier (GBC)</a> est une technique d'apprentissage automatique utilisée dans les problèmes de classification. Il s'agit d'une méthode d'ensemble où plusieurs modèles d'apprentissage faibles sont combinés pour former un modèle plus puissant.
            
            1. **`Initialisation`** : le modèle commence souvent par créer un arbre de décision peu profond, qui estime la classe cible.
            2. **`Calcul de l'erreur`** : Ensuite, le modèle calcule l’erreur entre sa prédiction et la vraie valeur cible.
            3. **`Réduction de l'erreur`** : L'objectif du Gradient Boosting Classifier est de réduire au maximum cette erreur, aussi appelée résidu, à l'aide de la technique de descente de gradient.
            4. **`Mise à jour des prédictions`** : Les prédictions de ce nouveau modèle sont ajoutées aux prédictions précédentes avec un certain facteur d'apprentissage (learning rate). Ainsi, chaque modèle corrige les erreurs résiduelles du modèle précédent.
            5. **`Répétition`** : Ce processus est répété plusieurs fois, chaque itération ajoutant un nouveau modèle qui se concentre sur les erreurs restantes des modèles précédents.
            6. **`Arrêt`** : Le modèle s'arrête soit sur un nombre fixe d'itérations, soit lorsque la performance sur un ensemble de validation cesse de s'améliorer, ou selon d'autres critères définis par l'utilisateur.
            """, unsafe_allow_html=True)
        
        else:
            lien_RFC = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
            
            
            st.markdown(f"""
            <a href="{lien_RFC}">Le Random Forest Classifier (RFC)</a> est une technique d'apprentissage automatique utilisée pour les problèmes de classification. C'est une méthode d'ensemble basée sur des arbres de décision.
            
            1. **`Construction des arbres de décision`** : Plusieurs arbres de décision sont construits de manière aléatoire - Les échantillons sont tirés avec remise, ce qui signifie qu’un même client peut être présent dans plusieurs arbres - Cela améliore la robustesse du modèle.
            
            2. **`Prédiction par vote à la majorité`**  : Une fois tous ces arbres créés, la prédiction est réalisée à l’aide d’un vote à la majorité - La classe qui obtient le plus grand nombre de votes devient la prédiction finale (0 ou 1) du Random Forest.
            
            3. **`Attention au surentraînement`** : Il est important de noter que ce genre de modèle est sensible au surentraînement.
            """, unsafe_allow_html=True)

    # if parameters:
    #     st.write('')
        
        
    if performance:
        
        # Affichage des scores
        
        st.write('')
        train_score = "{:.2f}".format(model.score(X_train, y_train))
        test_score = "{:.2f}".format(model.score(X_test, y_test))
        st.write(f"**Score** sur ensemble **train** &nbsp;&nbsp;&nbsp;&nbsp;**`{train_score}`**")
        st.write(f"**Score** sur ensemble **test** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**`{test_score}`**")
        st.write('')
        st.write('')     
        
        
        # Prédiction et rapport de classification   
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        st.code(f"Rapport de classification :\n{report}")
        st.write('')
        st.write('')        
        
        
        # Matrice de confusion sous forme de DataFrame (heatmap ne marche pas...)  
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        all_classes = np.array([0, 1])
        df_conf_matrix = pd.DataFrame(conf_matrix, index=[f'Classe {cls}' for cls in all_classes], columns=[f'Prédiction Classe {cls}' for cls in all_classes])
        
        # Fonction pour appliquer le style conditionnel
        def highlight_class(val):
            color = ''
            if val == conf_matrix[1, 1]:
                color = 'orange'
            elif val == conf_matrix[1, 0]:
                color = 'blue'
            return f'color: {color}'
        
        # Appliquer le style conditionnel
        styled_df = df_conf_matrix.style.applymap(highlight_class)
        
        # Afficher le DataFrame dans Streamlit
        st.write("Matrice de Confusion")
        st.dataframe(styled_df)
        st.write('') 
        st.write('') 


#Texte enregistré pour les checkbox Présentation des paramètres utilisés

description_hyper_gbc = """
- **`criterion`** : Critère de mesure de qualité de la séparation des arbres de décision dans le processus de boosting.
    - **friedman_mse** : Utilise l'erreur quadratique moyenne améliorée de Friedman, qui prend également en compte le gradient de la fonction de perte.
    - **squared_error** : Utilise simplement l'erreur quadratique moyenne, sans tenir compte du gradient. C'est plus rapide à calculer, mais potentiellement moins précis.                    
             
- **`learning_rate`** : Taux d'apprentissage, qui contrôle la contribution de chaque arbre au modèle global. Une valeur plus faible donne une meilleure généralisation, mais nécessite plus d'arbres dans l'ensemble.
- **`max_depth`** : Profondeur maximale de chaque arbre dans l'ensemble. Une profondeur plus grande permet au modèle de capturer des relations plus complexes dans les données, mais peut conduire à un surajustement si elle n'est pas régularisée.
- **`min_samples_split`** : Nombre minimum d'échantillons requis pour scinder un nœud interne. Cela régule la croissance de l'arbre en imposant des contraintes sur le nombre d'observations nécessaires pour former une nouvelle division.
- **`n_estimators`** : Nombre d'arbres dans l'ensemble. Un nombre plus élevé d'arbres peut améliorer les performances du modèle, mais cela nécessite également plus de temps de calcul.
- **`max_features`** : Nombre maximum de caractéristiques à considérer lors de la recherche de la meilleure division. Limiter le nombre de caractéristiques peut aider à réduire la variance et le temps de calcul, en particulier dans les ensembles de données avec de nombreuses fonctionnalités.
- **`subsample`** : Fraction des échantillons à utiliser pour l'entraînement de chaque arbre. Utiliser un sous-ensemble aléatoire des données pour chaque arbre peut aider à réduire la variance et à améliorer la généralisation.
"""
 
description_hyper_rfc = """
- **`criterion`** : Critère de mesure de qualité de la séparation des arbres de décision dans le processus de boosting.
    - **gini** : Utilise l'indice de Gini pour évaluer la pureté des nœuds de séparation.
    - **entropy** : Utilise l'entropie pour évaluer la pureté des nœuds de séparation.
    - **log_loss** : Spécifique aux problèmes de classification multi-classes et mesure l'erreur de logarithme.                   

- **`max_depth`** : Profondeur maximale de chaque arbre dans l'ensemble. Une profondeur plus grande permet au modèle de capturer des relations plus complexes dans les données, mais peut conduire à un surajustement si elle n'est pas régularisée.
- **`min_samples_split`** : Nombre minimum d'échantillons requis pour scinder un nœud interne. Cela régule la croissance de l'arbre en imposant des contraintes sur le nombre d'observations nécessaires pour former une nouvelle division.
- **`min_samples_leaf`** : le nombre minimum d'échantillons qu'un nœud terminal (ou feuille) doit contenir. Cela aide à empêcher la création de nœuds avec très peu d'échantillons, ce qui pourrait conduire à un modèle surajusté.
- **`n_estimators`** : Nombre d'arbres dans l'ensemble. Un nombre plus élevé d'arbres peut améliorer les performances du modèle, mais cela nécessite également plus de temps de calcul.
- **`max_features`** : Nombre maximum de caractéristiques à considérer lors de la recherche de la meilleure division. Limiter le nombre de caractéristiques peut aider à réduire la variance et le temps de calcul, en particulier dans les ensembles de données avec de nombreuses fonctionnalités.
- **`class_weight`** : Façon de traiter les déséquilibres de classe en donnant plus de poids aux classes moins fréquentes.
  - **balanced** : Ajuste automatiquement les poids des classes inversément proportionnels à leur fréquence.
  - **balanced_subsample** : Fonctionne de la même manière que 'balanced', mais ajuste également les poids à chaque échantillon d'arbre, plutôt qu'à l'ensemble de données global.
"""


# Sélection du modèle via liste déroulante
model_choice = st.selectbox(
    label='Sélectionner un modèle',
    options=['Gradient Boosting Classifier', 'Random Forest Classifier'], 
    index=None,  # Assurez-vous également que l'index est valide, 0 pour sélectionner le premier élément
    placeholder="Modèle . . .")  # Masquer le label tout en restant accessible


# Définition des modèles avant et après optimisation en fonction du choix de l'utilisateur
if model_choice:
    model_after = gbc_after if model_choice == 'Gradient Boosting Classifier' else rfc_after
    model_before = gbc_before if model_choice == 'Gradient Boosting Classifier' else rfc_before

    # Description du modèle
    with st.expander("Description du Modèle"):
        display_model_analysis(model_before, model_choice, description=True)
    
    # Paramètres du modèle
    with st.expander("Optimisation des hyperparamètres du Modèle"):
        
        if st.checkbox("Présentation des hyperparamètres utilisés", key='checkbox13'):
            st.write("""
<p style='font-size:18px'>Hyperparamètres du modèle <strong>{}</strong></p>
""".format(model_choice), unsafe_allow_html=True) 
            if model_choice == 'Gradient Boosting Classifier':
                st.markdown(description_hyper_gbc)
            else:
                st.markdown(description_hyper_rfc)
           
                
        if st.checkbox("Application de GridSearchCV()", key='checkbox11'):
            st.write("""
**`GridSearchCV est une technique d'optimisation des hyperparamètres qui effectue une recherche exhaustive sur un espace de paramètres spécifié. 
Elle utilise la validation croisée pour évaluer les performances de chaque combinaison de paramètres sur les données d'entrainement`**
""")        
            st.write("""
<p style='font-size:16px'>Grille des hyperparamètres pour le modèle <strong>{}</strong></p>
""".format(model_choice), unsafe_allow_html=True) 

            if model_choice == 'Gradient Boosting Classifier':
                st.table(gbc_param_grid_df)
            else:
                st.table(rfc_param_grid_df)
   
        
        if st.checkbox("Valeurs des Paramètres avant et après optimisation", key='checkbox12'):
            
            # Create two columns to display the model parameters
            col_before, col_after = st.columns(2)

            if model_choice == 'Gradient Boosting Classifier':
                with col_before:
                    st.write("""
        <p style='font-size:16px'><strong>Gradient Boosting Classifier par défaut<strong></p>
        """, unsafe_allow_html=True) 
                    st.table(params_gbc_before_df)

                with col_after:
                    st.write("""
        <p style='font-size:16px'><strong>Gradient Boosting Classifier Optimisé<strong></p>
        """, unsafe_allow_html=True) 
                    st.table(params_gbc_after_df)
            
            else:
  
                with col_before:
                    st.write("""
        <p style='font-size:16px'><strong>Random Forest Classifier par défaut<strong></p>
        """, unsafe_allow_html=True) 
                    st.table(params_rfc_before_df)

                with col_after:
                    st.write("""
        <p style='font-size:16px'><strong>Random Forest Classifier Optimisé<strong></p>
        """, unsafe_allow_html=True) 
                    st.table(params_rfc_after_df)
       
        
    # Performance du modèle avant et après optimisation
    
    with st.expander("Performance du Modèle avant et après optimisation des hyperparamètres"):
        st.write("""
<p style='font-size:16px'>Modèle <strong>{}</strong></p>
""".format(model_choice), unsafe_allow_html=True) 
        col1, col2 = st.columns(2)
        with col1:
            display_model_analysis(model_before, "Modèle par défaut", performance=True)
        with col2:
            display_model_analysis(model_after, "Modèle optimisé", performance=True)

        y_scores_before = model_before.predict_proba(X_test)[:, 1]
        y_scores_after = model_after.predict_proba(X_test)[:, 1]
        fpr_before, tpr_before, _ = roc_curve(y_test, y_scores_before)
        fpr_after, tpr_after, _ = roc_curve(y_test, y_scores_after)
        
        fig, ax = plt.subplots()
        ax.plot(fpr_before, tpr_before, label=f'ROC Modèle par défaut (AUC = {roc_auc_score(y_test, y_scores_before):.2f})', color='lightcoral')
        ax.plot(fpr_after, tpr_after, label=f'ROC Modèle optimisé (AUC = {roc_auc_score(y_test, y_scores_after):.2f})', color='lightblue')
        ax.set_title('Comparaison des Courbes ROC', fontsize=10, fontname='Arial', color='black')
        ax.set_xlabel('Taux de Faux Positifs', fontsize=9, fontname='Arial', color='black')
        ax.set_ylabel('Taux de Vrais Positifs', fontsize=9, fontname='Arial', color='black')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        # Définir les couleurs de fond et ajuster d'autres styles
        ax.set_facecolor('none')  # Correspond à plot_bgcolor='rgba(0,0,0,0)'
        fig.patch.set_facecolor('none')  # Correspond à paper_bgcolor='rgba(0,0,0,0)'
        
        # Afficher le graphique dans Streamlit
        st.pyplot(fig, use_container_width=True)

    
# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
#CSS pour que les boutons de la page (et surtout modélisation ) soient de la même largeur
st.markdown("""
    <style>
    .stButton button {
        width: 40%;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
if st.button("▶️\u2003 💡Interprétation_des_modèles"):
    st.switch_page("pages/5_💡_Interprétation_des_modèles.py")
    

# ------------------------------------------------------------------------------------------------

import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import time 


st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🎯"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

st.markdown('<h1 class="custom-title">Avez-vous souscrit ?</h1>', unsafe_allow_html=True)

if st.button("◀️\u2003 🎯 Recommandations métier - Conclusion"):
    st.switch_page("pages/6_🎯_Recommandations_métier_-_Conclusion.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------

# - Avez vous souscrit ? 
st.markdown("<h3 class='titre-h3'>Nous allons essayer de prédire votre choix...</h3>", unsafe_allow_html=True)
st.write("   ")
st.markdown("""
<ul>
Veuillez remplir ce questionnaire en fournissant des informations précises sur votre situation personnelle. 


Soyez assuré(e) qu'aucune donnée personnelle ne sera sauvegardée. 

Une fois complété, notre outil d'analyse déterminera si vous êtes susceptible d'être intéressé(e) par notre offre commerciale concernant le dépôt à terme.
</ul>
 """,unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------

X_train_standartscaller = pd.read_csv("Split_csv/3_bank_X_train_copie_before_standartscaller.csv",index_col=0)

def load_models():
    # Charger les modèles à partir des fichiers
    gbc_after = joblib.load("Models/model_gbc_after")
    rfc_after = joblib.load("Models/model_rfc_after")
    return gbc_after, rfc_after


def main():
    st.title("Formulaire de prédiction ")
    
    # Charger les modèles
    gbc_after, rfc_after = load_models()

    # Ajouter les champs du formulaire
    age = st.number_input(label = "Age",
                          min_value=18, max_value=90, step=1)
    job = st.selectbox(label= "Profession", 
                       options = ["admin", "technician", "services", "management", "retired", 
                                  "blue-collar", "unemployed", "housemaid", "self-employed", "student", "entrepreneur"])
    marital = st.selectbox(label = "État civil", 
                           options =["married", "single", "divorced"])
    education = st.selectbox(label= "Niveau scolaire", 
                             options = ["primary", "secondary","tertiary"])
    default = st.selectbox(label= "Avez vous un défault de paiement ?", 
                           options= ["yes", "no"])
    balance = st.number_input(label = "Quesl est votre solde annuel moyen ?",step=1)
    housing = st.selectbox(label= "Avez vous un prét immobilier en cour ?", 
                           options= ["yes", "no"])
    loan = st.selectbox(label= "Avez vous un prét personnel en cour ?", 
                        options= ["yes", "no"])
    month = st.selectbox(label = "Quand souhaitez vous etre contacté ?", 
                         options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = st.selectbox(label= "Quel était votre choix lors de la précédente campagne ? ", 
                           options= ["unknown", "failure", "success"])
    
    # chargement --------------------------------------------------------------------
    
    if st.button("En cliquant ici nous devinons votre choix"):
        with st.status("Télécharchement des données...", expanded=True) as status:
            st.write("Recherche des données...")
            time.sleep(2)
            st.write("Analyse des données...")
            time.sleep(2)
            st.write("Modélisation des données...")
            time.sleep(2)
        status.update(label="Nos modèles sont maintenant capable de prédire votre choix!", 
                      state="complete", 
                      expanded=False)
        
    # ------------------------------------------------------------------------------    
        # Créer un dictionnaire avec les valeurs
        client_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "month": month,
            "poutcome": poutcome,
            "pdays": 3,
            "day": 3,
            "duration" : 200, 
            "campaign" : 2,
        }
        
        # Convertir le dictionnaire en DataFrame
        df_new_clients = pd.DataFrame([client_data])

        # remplacement des valeurs catégorielles ordinales
        df_new_clients["education"] = df_new_clients["education"].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

        # remplacement des valeurs 'yes' et 'no' respectivement par '0' et '1'
        bin_cols = ["default", "housing", "loan"]
        df_new_clients[bin_cols] = df_new_clients[bin_cols].replace({'yes': 1, 'no': 0})
                
        # - month ----------------------------
        # j'encode ma variable que je stocke dans un df
        df_new_clients['month'] = pd.Categorical(df_new_clients['month'], categories=['apr', 'feb', 'mar', 'jan', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        encoded_month = pd.get_dummies(df_new_clients['month'], prefix='month_', drop_first=True)
        # je concat
        df_new_clients = pd.concat([df_new_clients, encoded_month], axis=1)
        # je supprime ma varible initiale
        df_new_clients = df_new_clients.drop("month", axis=1)

        # - marital ----------------------------
        df_new_clients['marital'] = pd.Categorical(df_new_clients['marital'], categories=["divorced", "single", "married"])
        encoded_marital = pd.get_dummies(df_new_clients['marital'], prefix='marital_', drop_first=True)
        df_new_clients = pd.concat([df_new_clients, encoded_marital], axis=1)
        df_new_clients = df_new_clients.drop("marital", axis=1)

        # - job ----------------------------
        df_new_clients['job'] = pd.Categorical(df_new_clients['job'], categories=["admin", "technician", "services", "management", "retired", 
                                  "blue-collar", "unemployed", "housemaid", "self-employed", "student", "entrepreneur"])
        encoded_job = pd.get_dummies(df_new_clients['job'], prefix='job_', drop_first=True)
        df_new_clients = pd.concat([df_new_clients, encoded_job], axis=1)
        df_new_clients = df_new_clients.drop("job", axis=1)

        # - poutcome ----------------------------
        df_new_clients['poutcome'] = pd.Categorical(df_new_clients['poutcome'], categories=["failure", "unknown", "success"])
        encoded_poutcome = pd.get_dummies(df_new_clients['poutcome'], prefix='poutcome_', drop_first=True)
        df_new_clients = pd.concat([df_new_clients, encoded_poutcome], axis=1)
        df_new_clients = df_new_clients.drop("poutcome", axis=1)
        
        # standardisation ------------------------------------------------------------------
        # Convertir toutes les colonnes en entiers
        cols = X_train_standartscaller.columns
        scaler = StandardScaler()
        X_train_standartscaller[cols] = scaler.fit_transform(X_train_standartscaller[cols])
        df_new_clients[cols] = scaler.transform(df_new_clients[cols])
        # -----------------------------------------------------------------------------------


        
        # Liste des colonnes dans l'ordre souhaité
        colonnes_ordre = [
            'age', 'education', 'default', 'balance', 'housing', 'loan', 'day', 'duration', 'campaign', 'pdays',
            'month__aug', 'month__dec', 'month__feb', 'month__jan', 'month__jul', 'month__jun', 'month__mar',
            'month__may', 'month__nov', 'month__oct', 'month__sep', 'marital__married', 'marital__single',
            'job__blue-collar', 'job__entrepreneur', 'job__housemaid', 'job__management', 'job__retired',
            'job__self-employed', 'job__services', 'job__student', 'job__technician', 'job__unemployed',
            'poutcome__success', 'poutcome__unknown'
            ]
        df_new_clients = df_new_clients[colonnes_ordre]

        # Prédire le résultat à l'aide des deux modèles
        prediction_gbc = gbc_after.predict(df_new_clients)
        prediction_rfc = rfc_after.predict(df_new_clients)

        # Afficher le DataFrame et les prédictions
        if prediction_gbc[0] == 1:
            st.markdown(
                """
                Notre modèle <span class="orange-bold">Gradient Boosting Classifier</span> prévoit que vous <span class="orange-bold">auriez</span> souscrit au dépôt
                """,unsafe_allow_html=True)
        else:
            st.markdown(
                """
                Notre modèle <span class="orange-bold">Gradient Boosting Classifier</span> prévoit que vous <span class="orange-bold">n'auriez pas</span> souscrit au dépôt
                """,unsafe_allow_html=True)
            
        # Affichage de la prédiction pour le modèle Random Forest Classifier
        if prediction_rfc[0] == 1:
            st.markdown(
                """
                Notre modèle <span class="orange-bold">Random Forest Classifier</span> prévoit que vous <span class="orange-bold">auriez</span> souscrit au dépôt
                """,unsafe_allow_html=True)
        else:
            st.markdown(
                """
                Notre modèle <span class="orange-bold">Random Forest Classifier</span> prévoit que vous <span class="orange-bold">n'auriez pas</span> souscrit au dépôt
                """,unsafe_allow_html=True)
            
        st.balloons()


if __name__ == "__main__":
    main()


st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")
if st.checkbox("Méthodologie de Prédiction", key='checkbox2'):

    st.markdown("""
<ul>
Pour obtenir ce résultat, nous basons notre analyse sur des clients d'une campagne précédente. Nous avons entraîné un modèle pour essayer de prédire votre choix. 

Cependant, nous avons dû anticiper et simuler certains paramètres. Par exemple, notre modèle est entraîné sur la durée des appels téléphoniques que les anciens clients ont eus avec notre service commercial. 

Étant donné que nous ne connaissons pas cette variable pour vous, nous l'avons choisie arbitrairement de manière à minimiser son influence sur le choix du modèle
</ul>
 """,unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003 💬 Contacts"):
    st.switch_page("pages/9_💬_Contacts.py")
# ------------------------------------------------------------------------------------------------










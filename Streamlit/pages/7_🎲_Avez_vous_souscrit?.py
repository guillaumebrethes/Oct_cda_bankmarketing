import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore


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

st.markdown('<h1 class="custom-title">Auriez vous souscrit/h1>', unsafe_allow_html=True)

if st.button("◀️\u2003💡 Interprétation des modèles"):
    st.switch_page("pages/5_💡_Interprétation_des_modèles.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------

# - Recommendation métier
st.markdown("<h3 class='titre-h3'>Auriez vous souscrit</h3>", unsafe_allow_html=True)
st.write("   ")
st.markdown("""
<ul>
    <li>Introduction du jeu </li>
</ul>
 """,unsafe_allow_html=True)

# - Conslusion 

st.markdown("<h3 class='titre-h3'>Conclusion</h3>", unsafe_allow_html=True)
st.write("   ")
st.markdown("""
    <ul>
        <li></li>
    </ul>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------


def load_models():
    # Charger les modèles à partir des fichiers
    gbc_after = joblib.load("Models/model_gbc_after")
    rfc_after = joblib.load("Models/model_rfc_after")
    return gbc_after, rfc_after


def main():
    st.title("Formulaire de Contact")
    
    # Charger les modèles
    gbc_after, rfc_after = load_models()

    # Ajouter les champs du formulaire
    prenom = st.text_input(label = "Prénom", 
                           max_chars=50)
    age = st.number_input(label = "Age",
                          min_value=0, max_value=150, step=1)
    job = st.selectbox(label= "Profession", 
                       options = ["admin", "technician", "services", "management", "retired", 
                                  "blue-collar", "unemployed", "housemaid", "self-employed", "student", "entrepreneur"])
    marital = st.selectbox(label = "État civil", 
                           options =["Married", "Single", "Divorced"])
    education = st.selectbox(label= "Niveau scolaire", 
                             options = ["primary", 
                                        "secondary",
                                        "tertiary"])
    default = st.selectbox(label= "Avez vous un défault de paiement ?", 
                           options= ["yes", "no"])
    balance = st.number_input(label = "Quesl est votre solde annuel moyen ?",step=1)
    housing = st.selectbox(label= "Avez vous un prét immobilier en cour ?", 
                           options= ["yes", "no"])
    loan = st.selectbox(label= "Avez vous un prét personnel en cour ?", 
                        options= ["yes", "no"])
    month = st.selectbox(label = "Quand souhaitez vous etre contacté ?", 
                         options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", ""])
    deposit = st.selectbox(label= "Avez vous un déja souscrit a une précédente campagne?", 
                           options= ["yes", "no"])
    poutcome = st.selectbox(label= "Quel était votre choix lors de la précédente campagne ? ", 
                           options= ["unknown", "failure", "success"])
    
    if st.button("En cliquant ici nous devinons votre choix"):
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
            "deposit": deposit,
            "poutcome": poutcome,
            "pday": 3,
            "duration" : 200
        }
        
        # Convertir le dictionnaire en DataFrame
        df_new_clients = pd.DataFrame([client_data])

        # Prédire le résultat à l'aide des deux modèles
        prediction_gbc = gbc_after.predict(df_new_clients.drop(columns=["Prénom"]))
        prediction_rfc = rfc_after.predict(df_new_clients.drop(columns=["Prénom"]))

        # Afficher le DataFrame et les prédictions
        st.write(df_new_clients)
        st.write(f"Prédiction du modèle Gradient Boosting Classifier : {prediction_gbc[0]}")
        st.write(f"Prédiction du modèle Random Forest Classifier : {prediction_rfc[0]}")


if __name__ == "__main__":
    main()















# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003 💬 Contacts"):
    st.switch_page("pages/9_💬_Contacts.py")
# ------------------------------------------------------------------------------------------------

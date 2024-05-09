import streamlit as st  # type: ignore

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="✅"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("Streamlit/styles.css")
# - - - - - - - 

st.markdown('<h1 class="custom-title">Interprétation des résultats</h1>', unsafe_allow_html=True)

if st.button("◀️\u2003📊 Modélisation"):
    st.switch_page("pages/4_⚙️_Modelisation.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)



# ------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
        /* Importation de la classe CSS pour les styles communs */
        .orange-bold { color: #EA6B14; font-weight: bold; }
    </style>
    <ul>
        <li>Au cours de ce projet, nous avons analysé les données issues de la récente campagne de télémarketing pour identifier les profils de clients les plus enclins à souscrire à des dépôts à terme. Cette démarche répond aux objectifs de la direction visant à accroître les bénéfices tout en minimisant les dépenses.</li><br>
        <li>Les modèles de machine learning que nous avons retenus, le <span class="orange-bold">Gradient Boost Classifier</span> et le <span class="orange-bold">Random Forest Classifier</span>, ont été adaptés pour maximiser le rappel de la classe 1. Ils ont démontré une <span class="orange-bold">amélioration notable</span> dans le ciblage des clients potentiels, avec un rappel de <span class="orange-bold">88,00 %</span> pour le <span class="orange-bold">Random Forest Classifier</span> et de <span class="orange-bold">86,09 %</span> pour le <span class="orange-bold">Gradient Boost Classifier</span>, réduisant ainsi efficacement le nombre de <span class="orange-bold">faux négatifs</span> et améliorant la performance de la campagne.</li><br>
        <li>A l'avenir, l'entreprise pourra s'appuyer sur ces modèles optimisés pour raffiner encore davantage ses stratégies de ciblage.</li><br>
        <li>Pour améliorer les résultats de la modélisation, la collecte de données supplémentaires sur les clients pourrait enrichir l'analyse et potentiellement augmenter la précision des modèles de prédiction. Nous pensons, entre autres, aux informations suivantes :
            <ul><br>
                <li><span class="orange-bold">Produits Financiers Actuels :</span> analyser la diversité des produits financiers que les clients possèdent déjà, pourrait nous aider à mieux comprendre leurs besoins et préférences financières.</li>
                <li><span class="orange-bold">Historique de la Fidélité à la Banque :</span> la durée de la relation d'un client avec la banque pourrait être un indicateur précieux de sa réceptivité aux nouvelles offres.</li>
                <li><span class="orange-bold">Satisfaction et interactions avec le Service Client :</span> les expériences précédentes des clients avec le service à la clientèle pourraient influencer leur disposition à souscrire à de nouveaux produits.</li>
            </ul>
        </li><br>
        <li>Ce projet, réalisé dans le cadre de notre formation en tant que data analysts, représente une opportunité significative pour mettre en pratique nos compétences et approfondir notre compréhension des techniques de machine learning. Grâce à cette expérience, nous nous sentons désormais plus confiants pour aborder des problématiques similaires dans nos futures missions professionnelles.</li>
    </ul>
    """, unsafe_allow_html=True)

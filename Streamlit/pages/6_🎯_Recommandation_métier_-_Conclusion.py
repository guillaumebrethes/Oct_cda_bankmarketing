import streamlit as st  # type: ignore

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

st.markdown('<h1 class="custom-title">Recommandation_métier - Conclusion</h1>', unsafe_allow_html=True)

if st.button("◀️\u2003💡 Interprétation des modèles"):
    st.switch_page("pages/5_💡_Interprétation_des_modèles.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------

# - Recommendation métier
st.markdown("<h3 class='titre-h3'>Recommandations métier</h3>", unsafe_allow_html=True)
st.write("   ")
st.markdown("""
<ul>
    <li>L'interprétation de ces deux modèles nous montre que ces derniers sont <span class="orange-bold">cohérents</span> & <span class="orange-bold">fiables</span> et que nous pouvons les déployer en environnement de production.</li><br>
    <li>Grâce à ces modèles, nous pouvons cibler les clients les plus susceptibles de souscrire à notre offre et gagner en efficacité financière et temporelle. Lors de la dernière campagne, sur le jeu test composé de <span class="orange-bold">1700 clients</span>, <span class="orange-bold">755</span> ont accepté l'offre, soit un taux de conversion de <span class="orange-bold">44,4 %</span>. L'utilisation de notre modèle pour sélectionner les candidats a réduit ce nombre à <span class="orange-bold">829</span>, parmi lesquels <span class="orange-bold">663</span> ont souscrit, portant le taux de conversion à près de <span class="orange-bold">80 %</span>. Cette approche stratégique maximise les résultats de la campagne tout en minimisant les efforts requis.</li><br>
    <li>Il est important de noter que la variable <span class="orange-bold">"duration"</span>, bien qu'étant un prédicteur significatif de la souscription, n'est pas connue avant l'initiation des appels en environnement de production. Les données suggèrent qu'un appel d'une durée supérieure à <span class="orange-bold">7 minutes</span> augmente significativement les chances de souscription. Un briefing détaillé et ciblé de l'équipe marketing sur ces constatations est indispensable.</li><br>
    <li>La variable <span class="orange-bold">"campaign"</span> indique également l'importance cruciale du premier contact, soulignant que les efforts doivent être concentrés pour maximiser les chances de souscription dès cette première interaction.</li><br>
    <li>Par ailleurs, la planification des campagnes devrait éviter les mois de <span class="orange-bold">Juillet, Août et Mai</span>, où une diminution de l'efficacité des campagnes est observée, vraisemblablement due aux vacances.</li><br>
    <li>La documentation complète des résultats des campagnes précédentes est primordiale. Nos analyses indiquent que les clients ayant antérieurement réagi positivement sont plus susceptibles de réitérer leur engagement lors de campagnes futures.</li><br>
    <li>Ces recommandations stratégiques, basées sur notre analyse de données et l'interprétation des modèles, sont cruciales pour améliorer l'efficacité des campagnes futures et maximiser le retour sur investissement, tout en reconnaissant les défis posés par des variables comme <span class="orange-bold">"duration"</span> et <span class="orange-bold">"campaign"</span> qui ne sont pas préalablement connues.</li>
</ul>
 """,unsafe_allow_html=True)

# - Conslusion 

st.markdown("<h3 class='titre-h3'>Conclusion</h3>", unsafe_allow_html=True)
st.write("   ")
st.markdown("""
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


# ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.write("   ")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.write("   ")

if st.button("▶️\u2003 🎲 Avez vous souscrit ?"):
    st.switch_page("pages/7_🎲_Avez_vous_souscrit?.py")
# ------------------------------------------------------------------------------------------------

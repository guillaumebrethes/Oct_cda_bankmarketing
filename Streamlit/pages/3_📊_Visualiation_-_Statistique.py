import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns # type: ignore
import statsmodels.api # type: ignore
from plotly import graph_objs as go # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
import plotly.figure_factory as ff # type: ignore
from scipy.stats import chi2_contingency # type: ignore



# Variables 
df = pd.read_csv("bank.csv")

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="📊"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 

# titre
st.markdown('<h1 class="custom-title">Visualisation et Statistique</h1>', unsafe_allow_html=True)


if st.button("◀️\u2003📖 Présentation - Exploration"):
    st.switch_page("pages/2_📖_Présentation_-_Exploration.py")
    
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)

st.markdown(
    """
    Dans ce chapitre nous allons étudier plus en profondeur notre jeu de données.

    Nous allons aborder l'étude selon 2 axes principaux :

    - **La visualisation** à l'aide de graphiques pertinents\n\n
    - **L'étude statistique** pour corroborer notre exploration et visualisation
    """)


#--------------------------------------------------------------------------------------------
# Affichage de la repartition de la variable Deposit en camembert 
#--------------------------------------------------------------------------------------------
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Visualisation de la variable cible</h3>", unsafe_allow_html=True)


a = df.groupby(['deposit'],
        as_index= False)['age'].count().rename(columns= {'age':'Count'})

a['percent'] = round(a['Count'] * 100 / a.groupby('deposit')['Count'].transform('sum'),1)
a['percent'] = a['percent'].apply(lambda x: '{}%'.format(x))

figdeposit = px.pie(a, 
                    values='Count', 
                    names='deposit', 
                    color='deposit',
                    width=600, 
                    height=450,
                    color_discrete_sequence= ['lightcoral', 'lightblue'],
                    hole=0.3)
figdeposit.update_traces(text=a['percent'], textposition='inside', textinfo='percent+label')
st.plotly_chart(figdeposit)

st.markdown(
    """
    La répartition des données concernant la variable cible <span class="orange-bold">deposit</span> est relativement équilibrée, ce qui représente un atout pour la modélisation.
    """, unsafe_allow_html=True)


#--------------------------------------------------------------------------------------------
# Affichage des caractéristiques socio démographiques des clients
#--------------------------------------------------------------------------------------------
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Caractéristiques socio-démographiques des clients</h3>", unsafe_allow_html=True)

# Selection du graphique à afficher
st.write("   ")
graph_choisi_socio = st.selectbox(
    label="Sélectionner les variables à étudier", 
    options=["Age en fonction de Deposit",
             "Job en fonction de Deposit",
             "Marital en fonction de Deposit",
             "Education en fonction de Deposit"],
    index=None, 
    placeholder=". . .")

#---------------------------------------
# Age 
if graph_choisi_socio == 'Age en fonction de Deposit':
    
    # Graphique 
    x1 = df[df['deposit'] == 'yes']['age']
    x2 = df[df['deposit'] == 'no']['age']
    hist_data = [x1, x2]
    group_labels = ['Deposit = yes', 'Deposit = no']
    colors = ['skyblue', 'lightcoral']
    density_fig = ff.create_distplot(hist_data,
                                     group_labels,
                                     colors= colors,
                                     show_rug= False,
                                     bin_size= 1,
                                     show_hist= False)
    
    st.markdown("#### 📊 Visualisation")

    density_fig.update_layout(title= '<b style="color:black; font-size:90%;">Distribution des âges</b>',
                              xaxis_title= 'Âge',
                              yaxis_title= 'Densité')
    st.plotly_chart(density_fig)

    # Statistique
    st.markdown("#### 📈 Statistique")

    resultAD = statsmodels.formula.api.ols('age ~ deposit', data=df).fit()
    tableAD = statsmodels.api.stats.anova_lm(resultAD)
    
    st.markdown("""
    **Test ANOVA :** \n\n
    Valeur de la p-valeur : `{}`
    
    #### 💬 Interprétation 

    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05. 
    
    L'âge minimum des clients de notre jeu de données est de **18 ans** et le maximum est de **95 ans**.\n\n
    La majorité des clients de notre jeu de données ont entre **32** et **49 ans**.\n\n
    On remarque que pour les plus de **58 ans** et les moins de **30 ans**, la part de clients qui ont souscrit au dépôt à terme est plus importante.
""".format(tableAD['PR(>F)']['deposit']))

#---------------------------------------
# Job 
elif graph_choisi_socio == 'Job en fonction de Deposit':

    # Graphique     

    b = df.groupby(['job','deposit'],
                   as_index= False)['age'].count().rename(columns= {'age':'Count'})
    b['percent'] = round(b['Count'] * 100 / b.groupby('job')['Count'].transform('sum'),1)
    b['percent'] = b['percent'].apply(lambda x: '{}%'.format(x))
    
    st.markdown("#### 📊 Visualisation")
    figjob = px.bar(b,
                    x= 'job',
                    y= 'Count',
                    text= 'percent',
                    color= 'deposit',
                    barmode= 'group',
                    color_discrete_sequence= ['lightcoral', 'lightblue'],
                    width= 600, height= 450)
        
    figjob.update_traces(marker= dict(line= dict(color= '#000000', width= 1)), 
                         textposition = "outside")
    
    figjob.update_layout(showlegend= True,
                         title_text= '<b style="color:black; font-size:90%;">Distribution des job en fonction de deposit</b>',
                         font_family= "Arial",
                         title_font_family= "Arial")
    st.plotly_chart(figjob)
             
    # Statistique
    st.markdown("#### 📈 Statistique")  
      
    ctDJ = pd.crosstab(df['deposit'], df['job'])
    resultats_chi2DJ = chi2_contingency(ctDJ)
    
    st.markdown("""
    **- Test KI-deux** :
    
    Résultat statistique : `{}`
    
    Résultat p_valeur : `{}`
    
    #### 💬 Interprétation 
    
    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05.
    Les professions **Management (22,8%)**, **Blue Collar(17,1%)** et **Technicians(16,9%)** sont les plus représentées dans notre jeu de   données.

    Lorsque l'on regarde pour chaque catégorie la différence entre la souscription ou non au dépôt à terme, on constate que pour les    professions: **retired** et **student** ont une tendance à plus souscrire à ce produit bancaire.

    A la différence des professions **blue-collar**, **services** et **technician** qui ont une valeur à **Non** supérieur à **Oui.**
    """.format(resultats_chi2DJ[0], resultats_chi2DJ[1]))

#---------------------------------------
# Marital
elif graph_choisi_socio == 'Marital en fonction de Deposit' :
    
    # Graphique
    c = df.groupby(['marital','deposit'],
                   as_index= False)['age'].count().rename(columns={'age':'Count'})
    c['percent'] = round(c['Count'] * 100 / c.groupby('marital')['Count'].transform('sum'), 1)
    c['percent'] = c['percent'].apply(lambda x: '{}%'.format(x))
    
    st.markdown("#### 📊 Visualisation")

    figmarital = px.bar(c,
                        x= 'marital',
                        y= 'Count',
                        text= 'percent',
                        color= 'deposit',
                        barmode= 'group',
                        color_discrete_sequence= ['lightcoral','lightblue'],
                        width= 600, 
                        height= 450)
    figmarital.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")
        
    figmarital.update_layout(showlegend=True,
                             title_text='<b style="color:black; font-size:90%;">Distribution de marital en fonction de deposit</b>',
                             font_family="Arial",
                             title_font_family="Arial")
    st.plotly_chart(figmarital)
    
    # Statistique
    st.markdown("#### 📈 Statistique") 

    ctDM = pd.crosstab(df['deposit'], df['marital'])
    resultats_chi2DM = chi2_contingency(ctDM)
    st.markdown("""
    **- Test KI-deux** :
    
    Résultat statistique : `{}`
    
    Résultat p_valeur : `{}`

    #### 💬 Interprétation 

    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05.

    Plus de la moitié des clients de notre jeu de données sont **mariés** et **(56.9%)** d'entre eux ne souscrivent pas au dépôt.
    
    Les **célibataires** ont proportionnellement plus souscrit au dépôt à terme que les clients **mariés**.
    """.format(resultats_chi2DM[0], resultats_chi2DM[1]))
    
#---------------------------------------
# Education
elif graph_choisi_socio == 'Education en fonction de Deposit' :
    
    # Graphique
    st.markdown("#### 📊 Visualisation")
    
    d = df.groupby(['education','deposit'],
                   as_index= False)['age'].count().rename(columns= {'age':'Count'})
    d['percent'] = round(d['Count'] * 100 / d.groupby('education')['Count'].transform('sum'), 1)
    d['percent'] = d['percent'].apply(lambda x: '{}%'.format(x))
        
    figeducation = px.bar(d,
                          x= 'education',
                          y= 'Count',
                          text= 'percent',
                          color= 'deposit',
                          barmode= 'group',
                          color_discrete_sequence= ['lightcoral', 'lightblue'],
                          width= 600, 
                          height= 450)
    
    figeducation.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")
        
    figeducation.update_layout(showlegend=True,
                               title_text='<b style="color:black; font-size:90%;">Distribution de education en fonction de deposit</b>',
                               font_family="Arial",
                               title_font_family="Arial")
    st.plotly_chart(figeducation)
        
    # Statistique   
    st.markdown("#### 📈 Statistique")  
    ctDM = pd.crosstab(df['deposit'], df['education'])
    resultats_chi2DM = chi2_contingency(ctDM)
    
    st.markdown("""
    **- Test KI-deux** :
    
    Résultat statistique : `{}`
    
    Résultat p_valeur : `{}`

    #### 💬 Interprétation 
    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05.
    
    **(51,2%)** des clients ont un niveau d'études secondaires et **(34.9%)** un niveau d'études supérieures.
    
    On remarque que les clients avec un niveau d'études supérieures **('tertiary')** représentent la catégorie qui a le plus souscrit au dépôt à terme par rapport aux 2 autres.
    """.format(resultats_chi2DM[0], resultats_chi2DM[1]))

#--------------------------------------------------------------------------------------------
# Affichage des caractéristiques bancaires des clients
#--------------------------------------------------------------------------------------------
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
st.markdown("<h3 class='titre-h3'>Caractéristiques bancaires des clients</h3>", unsafe_allow_html=True)


graph_choisi_banc = st.selectbox(label="Selectionner les variables à étudier", 
                                 options=["Default en fonction de Deposit",
                                          "Housing en fonction de Deposit",
                                          "Loan en fonction de Deposit",
                                          "Balance en fonction de Deposit"], 
                                 index=None,
                                 placeholder=". . .")

#---------------------------------------
# Default 
if graph_choisi_banc == 'Default en fonction de Deposit' :
    st.write("  ")

#---------------------------------------
# Housing
elif graph_choisi_banc == 'Housing en fonction de Deposit' :

    # Graphique
    st.markdown("#### 📊 Visualisation")
    
    e = df.groupby(['housing','deposit'],
                   as_index=False)['age'].count().rename(columns={'age':'Count'})
    e['percent'] = round(e['Count'] * 100 / e.groupby('housing')['Count'].transform('sum'),1)
    e['percent'] = e['percent'].apply(lambda x: '{}%'.format(x))
    fighousing = px.bar(e,
                    x= 'housing',
                    y= 'Count',
                    text= 'percent',
                    color= 'deposit',
                    barmode= 'group',
                    color_discrete_sequence= ['lightcoral', 'lightblue'],
                    width= 600, height= 450)

    fighousing.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")
    fighousing.update_layout(showlegend=True,
                             title_text='<b style="color:black; font-size:90%;">Distribution de Housing en fonction de deposit</b>',font_family="Arial",
                             title_font_family="Arial")
    st.plotly_chart(fighousing)

# Statistique   
    st.markdown("#### 📈 Statistique") 
    ctDH = pd.crosstab(df['deposit'], df['housing'])
    resultats_chi2DH = chi2_contingency(ctDH)
    st.markdown("""
    **- Test KI-deux** :
    
    Résultat statistique : `{}`
    
    Résultat p_valeur : `{}`

    #### 💬 Interprétation 
    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05.

    Notre jeu de données a une répartition assez équilibrée sur cette variable, **52%** ont un prêt immobilier contre **48%** sans prêt     immobilier.
    Le graphique ci-contre nous montre que les clients qui ont un prêt immobilier ont tendance à ne pas souscrire au dépôt à terme **(63.3%)**.
    """.format(resultats_chi2DH[0], resultats_chi2DH[1]))



#---------------------------------------
# Loan
elif graph_choisi_banc == 'Loan en fonction de Deposit' :
    
    f = df.groupby(['loan','deposit'],
    as_index=False)['age'].count().rename(columns={'age':'Count'})
    f['percent'] = round(f['Count'] * 100 / f.groupby('loan')['Count'].transform('sum'),1)
    f['percent'] = f['percent'].apply(lambda x: '{}%'.format(x))
    
# Graphique
    st.markdown("#### 📊 Visualisation")
    figloan = px.bar(f,
                     x= 'loan',
                     y= 'Count',
                     text= 'percent',
                     color= 'deposit',
                     barmode= 'group',
                     color_discrete_sequence= ['lightcoral', 'lightblue'],
                     width=600, height=450)

    figloan.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")
    figloan.update_layout(showlegend=True,
                          title_text='<b style="color:black; font-size:90%;">Distribution de Loan en fonction de deposit</b>',font_family="Arial",
                          title_font_family="Arial")
    st.plotly_chart(figloan)

# Statistique   
    st.markdown("#### 📈 Statistique") 
    ctDL = pd.crosstab(df['deposit'], df['loan'])
    resultats_chi2DL = chi2_contingency(ctDL)
    st.markdown("""
    **- Test KI-deux** :
    
    Résultat statistique : `{}`
    
    Résultat p_valeur : `{}`

    #### 💬 Interprétation 
    Le test nous montre qu'il y a une relation entre les deux variables, car la valeur de la p-valeur est inférieur à 0,05.

    Notre jeu de données a une répartition déséquilibrée sur cette variable, **86.5%** des clients n'ont pas de prêt personnel contre **13.5%**     qui en ont un.
    
    Le graphique ci-dessus nous montre que les clients qui ont un prêt personnel ont tendance à ne pas souscrire au dépôt à terme **(67%)**.
    """.format(resultats_chi2DL[0], resultats_chi2DL[1]))

    
    
#---------------------------------------
# Balance
elif graph_choisi_banc == 'Balance en fonction de Deposit' :

# Graphique
    st.markdown("#### 📊 Visualisation")
    
    # Définition des valeurs initiales
    balance_min_default = -2500
    balance_max_default = 15000
    nbins_default = 100

    # Définition du slider pour la plage de valeurs de balance
    balance_range = st.slider("Plage de valeurs pour balance", 
                          min_value=-5000, 
                          max_value=20000, 
                          value=(balance_min_default, balance_max_default))

    # Extraction des valeurs min et max de la plage
    balance_min = balance_range[0]
    balance_max = balance_range[1]

    # Définition du slider pour nbins
    nbins = st.slider("Nombre de bacs (bins)", min_value=10, max_value=200, value=nbins_default)

    # Filtrage des données en fonction de la plage de valeurs
    balance_filtre = df[(df['balance'] >= balance_min) & (df['balance'] <= balance_max)]

    # Création du graphique
    fig = px.histogram(balance_filtre, 
                   x='balance',
                   color='deposit', 
                   marginal='box', 
                   hover_data=df.columns, 
                   color_discrete_map={'yes': 'lightblue', 'no': 'lightcoral'}, 
                   nbins=nbins)
    fig.update_layout(yaxis_title="Nombre de clients")

    # Affichage du graphique
    st.plotly_chart(fig)
    
    # Couleur bulle slider
    st.markdown("""
    <style>
    .st-emotion-cache-1vzeuhh {background-color: black;}
    </style>
    """, unsafe_allow_html=True)



    
# Statistique   
    st.markdown("#### 📈 Statistique") 
    resultBD = statsmodels.formula.api.ols('balance ~ deposit', data=df).fit()
    tableBD = statsmodels.api.stats.anova_lm(resultBD)
    
    st.markdown("""
    **-Test ANOVA** : PR(>F) : `{}`
    
    #### 💬 Interprétation 
    Le test nous montre qu'il y a une relation entre les deux variables

    Sur ces graphiques, nous avons décidé de restreindre les valeurs des abscisses **(balance)** afin de leur donner plus de clarté. La     distribution de balance est très écrasée à cause de valeurs extrêmes très grandes (maximum est égal à 81204).
    En visualisant les graphiques Boîtes à moustache ci-dessus, on peut voir que les clients qui souscrivent au dépôt à terme ont un solde moyen    plus élevé que ceux qui ne souscrivent pas.
    """.format(tableBD['PR(>F)']['deposit']))

            
            
with st.expander(label="Autres tests", expanded=False):
    col1,col2,col3=st.columns(3)
    with col1:
        ctLH = pd.crosstab(df['loan'], df['housing'])
        resultats_chi2LH = chi2_contingency(ctLH)
        st.write("Test KI-deux housing/loan :")
        st.write("Résultat statistique :",resultats_chi2LH[0])
        st.write("Résultat p_valeur :",resultats_chi2LH[1])
    with col2:
        ctBH = pd.crosstab(df['balance'], df['housing'])
        resultats_chi2BH = chi2_contingency(ctBH)
        st.write("Test KI-deux housing/balance :")
        st.write("Résultat statistique :",resultats_chi2BH[0])
        st.write("Résultat p_valeur :",resultats_chi2BH[1])
    with col3:
        resultBL = statsmodels.formula.api.ols('balance ~ loan', data=df).fit()
        tableBL = statsmodels.api.stats.anova_lm(resultBL)
        st.write("Test ANOVA balance/loan :")
        st.write("PR(>F) :",tableBL['PR(>F)']['loan'])
        st.write("   ")
        st.write("Ces variables sont toutes liées entre elles, cependant elles apportent chacune des informations bien différentes les unes des autres il est donc pertinent de toutes les garder dans le jeu de données.")
    #--
    
    
    
    
    # ------------------------------------------------------------------------------------------------
# bouton de basculement de page 
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)
if st.button("▶️\u2003 ⚙️ Modelisation"):
    st.switch_page("pages/4_⚙️_Modelisation.py")
# ------------------------------------------------------------------------------------------------
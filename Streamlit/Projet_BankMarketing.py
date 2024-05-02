import streamlit as st
import numpy as np
#%matplotlib inline
import pandas as pd
import seaborn as sns
import calendar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
from plotly import graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency

# Chargement du fichier de données
df = pd.read_csv("bank.csv")

# Mise en place d'un titre pour le projet
#st.markdown("<h3 style='font-size: 20px; font-family: Arial, sans-serif;'>Prédiction du succès d'une campagne de Marketing d'une banque</h>", unsafe_allow=True)
st.write("Prédiction du succès d'une campagne de Marketing d'une banque")

# Configuration de la barre 
with st.sidebar:
    st.image("Sommaire.jpg")
    st.title("Sommaire")
    pages = ["Contexte du projet",
             "Présentation des données",
             "Exploration du jeu de données",
             "Visualisations et statistiques",
             "Pré-processing",
             "Classification du problème",
             "Modélisation", 
             "Interprétation des résultat",
             "Recommendation métier"]
    page = st.radio("Aller vers la page :",pages)
    st.markdown("**FORMATION**")
    st.write("Data Analyst Octobre 2023 - Juin 2024")
    st.markdown("**AUTEURS**")
    st.write("Morgane Coateval")
    st.write("Estelle Mir-Emarati")
    st.write("Guillaume Brethes")
    st.write("Laurent Cochard")

# Traitement de la page 1
if page == pages[0] :
    st.write("### PROJET ###")
    st.write("Prédiction du succès d'une campagne de Marketing d'une banque")
    st.write("### Contexte du projet ###")
    st.write("Ce projet s'inscrit dans le cadre de l'utilisation des sciences des données appliquées dans les")
    st.write("entreprises de service et plus précisément dans le domaine bancaire. Au sein du secteur bancaire,")
    st.write("l'optimisation du ciblage du télémarketing est un enjeu clé, sous la pression croissante d'augmenter")
    st.write("les profits et de réduire les coûts.")
    st.write("Nous avons à disposition les données de la dernière campagne télémarketing d'une banque pour la ")
    st.write("vente de dépôts à terme. Ce jeu de données est en accès libre sur la plateforme Kaggle.com.")
    st.write("L'objectif est de prédire quels clients sont les plus susceptibles de souscrire au dépôt à terme.")
    st.image("banque.jpg")

# Traitement de la page 1
elif page == pages[1]:
    st.write("### Présentation des données ###")
    #---------------------------------------
    if st.checkbox("Contenu du Dataset") :
        st.markdown("Nous avons un jeu de données qui se compose de **11 162 lignes et 17 colonnes**, il contient des")
        st.write("valeurs numériques ainsi que des valeurs textuelles. Vous pouvez visualiser les premières lignes de")
        st.markdown("celui-ci ci-dessous. Dans ce jeu de données nous avons des informations sur les **caractéristiques**")
        st.markdown("**socio-démographiques** *(âge, type d'emploi, situation familiale, niveau d'études)* et **bancaires des**")
        st.markdown("**clients** *(solde moyen du compte, prêt immobilier en cours, autres prêts en cours)* ainsi que des ")
        st.markdown("informations sur **les caractéristiques de la campagne** *(Durée du dernier appel, nombre de contacts*")
        st.write("sur la campagne, nombre de contacts avant cette campagne, le nombre de jours écoulés depuis la")
        st.write("dernière campagne et le résultat de la campagne).")
        st.dataframe(df.head())
        if st.checkbox("Afficher les doublons") :
            st.write(df.duplicated().sum())
        if st.checkbox("Afficher les valeurs manquantes") :
            st.dataframe(df.isna().sum())
    #---------------------------------------
    if st.checkbox("Variable cible") :
        st.write("La variable cible est 'deposit' qui est un produit appelé 'dépôt à terme'. C'est un produit souscrit")
        st.write("par le client qui dépose une somme d'argent qui sera bloquée sur une période. Dans le jeu de")
        st.markdown("données elle se répartie en deux valeurs **'Yes et No'**. Avant le nettoyage du jeu de données, la")
        st.markdown("distribution de la variable est assez équilibrée **52.6%** pour **No** et **47.4%** pour **Yes**. Cette distribution")
        st.write("équilibrée nous permettra d'éviter un potentiel biais d'entraînement dans la modélisation.")
    #---------------------------------------
    if st.checkbox("Tableau explicatif des variables") :
        st.write("Tableau des données")
        st.image("Description_donnees.jpg")

# Traitement de la page 2
elif page == pages[2]:
    st.write("### Exploration du jeu de données ###")
    st.write("2.1 Gestion des valeurs non désirées ou manquantes")
    st.image("Valeurs_non_desirees.jpg")

# Traitement de la page 3
elif page == pages[3]:
    st.write("### Visualisations et statistiques ###")
    #---------------------------------------
    # Affichage de la repartition de la variable Deposit en camembert 
    if st.checkbox("Variable cible: Deposit") :
        donnees=df['deposit'].value_counts()
        col=['no','yes']
        figdeposit=plt.pie(donnees,labels=col,colors= ['lightcoral', 'lightblue'],autopct='%1.1f%%',explode= (0.05, 0.05))
        plt.title('Répartition des souscriptions au dépôt à terme', fontweight='bold')
        st.plotly_chart(figdeposit)
    #---------------------------------------
    # Affichage des caractéristiques socio démographiques des clients
    if st.checkbox("Caractéristiques socio-démographiques des clients") :
        # Selection du graphique à afficher
        graphchoisi=st.selectbox(label="Vue", options=["Age en fonction de Deposit","Job en fonction de Deposit","Marital en fonction de Deposit","Education en fonction de Deposit"])
        if graphchoisi == 'Age en fonction de Deposit' :
        # Affichage des courbe de la variable Deposit en fonction de l'age
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

            density_fig.update_layout(title= '<b style="color:black; font-size:110%;">Distribution des âges</b>',
                        xaxis_title= 'Âge',
                        yaxis_title= 'Densité')

            st.plotly_chart(density_fig)
        #---------------------------------------
        elif graphchoisi == 'Job en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Job
            b = df.groupby(['job','deposit'],
              as_index= False)['age'].count().rename(columns= {'age':'Count'})

            b['percent'] = round(b['Count'] * 100 / b.groupby('job')['Count'].transform('sum'),1)
            b['percent'] = b['percent'].apply(lambda x: '{}%'.format(x))

            figjob = px.bar(b,
                x= 'job',
                y= 'Count',
                text= 'percent',
                color= 'deposit',
                barmode= 'group',
                color_discrete_sequence= ['lightcoral', 'lightblue'],
                width= 600, height= 450)

            figjob.update_traces(marker= dict(line= dict(color= '#000000', width= 1)), textposition = "outside")

            figjob.update_layout(title_x= 0.5,
                  showlegend= True,
                  title_text= '<b style="color:black; font-size:110%;">Distribution des job en fonction de deposit</b>',
                  font_family= "Arial",
                  title_font_family= "Arial")
            st.plotly_chart(figjob)
        #---------------------------------------
        elif graphchoisi == 'Marital en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Marital
            c = df.groupby(['marital','deposit'],
              as_index= False)['age'].count().rename(columns={'age':'Count'})

            c['percent'] = round(c['Count'] * 100 / c.groupby('marital')['Count'].transform('sum'), 1)
            c['percent'] = c['percent'].apply(lambda x: '{}%'.format(x))

            figmarital = px.bar(c,
                x= 'marital',
                y= 'Count',
                text= 'percent',
                color= 'deposit',
                barmode= 'group',
                color_discrete_sequence= ['lightcoral','lightblue'],
                width= 600, height= 450)

            figmarital.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")

            figmarital.update_layout(title_x=0.5,
                  showlegend=True,
                  title_text='<b style="color:black; font-size:110%;">Distribution de marital en fonction de deposit</b>',
                  font_family="Arial",
                  title_font_family="Arial")
            st.plotly_chart(figmarital)          
        #---------------------------------------
        elif graphchoisi == 'Education en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Education
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
                width= 600, height= 450)

            figeducation.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")

            figeducation.update_layout(title_x=0.5,
                  showlegend=True,
                  title_text='<b style="color:black; font-size:110%;">Distribution de education en fonction de deposit</b>',
                  font_family="Arial",
                  title_font_family="Arial")
            st.plotly_chart(figeducation)          
    #---------------------------------------          
    if st.checkbox("Caractéristiques bancaires des clients") :
        # Selection du graphique à afficher
        graphchoisi=st.selectbox(label="Vue", options=["Default en fonction de Deposit","Housing en fonction de Deposit","Loan en fonction de Deposit","Balance en fonction de Deposit"])
        # Affichage du graphique de la variable Deposit en fonction de Default
        if graphchoisi == 'Default en fonction de Deposit' :
            st.write("toto")
        #---------------------------------------
        elif graphchoisi == 'Housing en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Housing
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
            fighousing.update_layout(title_x=0.5,
                  showlegend=True,
                  title_text='<b style="color:black; font-size:110%;">Distribution de Housing en fonction de deposit</b>',
                  font_family="Arial",
                  title_font_family="Arial")
            st.plotly_chart(fighousing)  
        #---------------------------------------
        elif graphchoisi == 'Loan en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Loan
            f = df.groupby(['loan','deposit'],
               as_index=False)['age'].count().rename(columns={'age':'Count'})

            f['percent'] = round(f['Count'] * 100 / f.groupby('loan')['Count'].transform('sum'),1)
            f['percent'] = f['percent'].apply(lambda x: '{}%'.format(x))

            figloan = px.bar(f,
                x= 'loan',
                y= 'Count',
                text= 'percent',
                color= 'deposit',
                barmode= 'group',
                color_discrete_sequence= ['lightcoral', 'lightblue'],
                width=600, height=450)

            figloan.update_traces(marker=dict(line=dict(color='#000000', width=1)),textposition = "outside")
            figloan.update_layout(title_x=0.5,
                  showlegend=True,
                  title_text='<b style="color:black; font-size:110%;">Distribution de Loan en fonction de deposit</b>',
                  font_family="Arial",
                  title_font_family="Arial")
            st.plotly_chart(figloan)
        #---------------------------------------
        elif graphchoisi == 'Balance en fonction de Deposit' :
        # Affichage du graphique de la variable Deposit en fonction du Balance
            x0 = df[df['deposit'] == 'yes']['balance']
            x1 = df[df['deposit'] == 'no']['balance']

            figdeposit = go.Figure()
            figdeposit.add_trace(go.Box(x=x0,
                     name= 'deposit_yes',
                     marker_color= 'lightblue'))
            figdeposit.add_trace(go.Box(x=x1,
                     name= 'deposit_no',
                     marker_color= 'lightcoral'))

            figdeposit.update_layout(width= 700, height= 470, showlegend= False)
            figdeposit.update_layout(title_x= 0.5,
                  title_text= '<b style="color:black; font-size:110%;">Distribution de Balance en fonction de deposit </b>',
                  font_family= "Arial",
                  title_font_family= "Arial")
            figdeposit.update_traces(orientation='h')
            st.plotly_chart(figdeposit)
    #---------------------------------------
    if st.checkbox("Caractéristiques de la campagne marketing") :
        st.write("Toto")

# Traitement de la page 4
elif page == pages[4]:
    st.write("### Pré-processing ###")

# Traitement de la page 5
elif page == pages[5]:
    st.write("### Classification du problème ###")

# Traitement de la page 6
elif page == pages[6]:
    st.write("### Modélisation ###")

# Traitement de la page 7
elif page == pages[7]:
    st.write("### Interprétation des résultat ###")

# Traitement de la page 1
elif page == pages[8]:
    st.write("### Recommendation métier ###")
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px # type: ignore
import plotly.figure_factory as ff # type: ignore
import statsmodels.api # type: ignore
from scipy.stats import chi2_contingency # type: ignore



# Variables 
df = pd.read_csv("/Users/gub/Documents/Privé/Formations/DataScientest/Data_projet/Oct_cda_bankmarketing/Streamlit/pages/bank.csv")

# Page
st.set_page_config(
    page_title="Bank Marketing",
    page_icon="🔍" 
)

st.title("Exploration du jeu de données")

if st.button("◀️\u2003📖 Présentation - Exploration"):
    st.switch_page("pages/2_📖_Presentation_-_Exploration.py")
st.write("---")

st.markdown("""
Dans ce chapitre nous allons étudier plus en profondeur notre jeu de données.

Nous allons aborder l'étude selon 2 axes principaux :

- **La visualisation** à l'aide de graphiques pertinents\n\n
- **L'étude statistique** pour corroborer notre exploration et visualisation
""")


#--------------------------------------------------------------------------------------------
# Affichage de la repartition de la variable Deposit en camembert 
#--------------------------------------------------------------------------------------------
st.write("---")
st.write("### Visualisation de la variable cible ###")

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

st.write("A completer")

#--------------------------------------------------------------------------------------------
# Affichage des caractéristiques socio démographiques des clients
#--------------------------------------------------------------------------------------------
st.write("---")
st.write("### Caractéristiques socio-démographiques des clients ###")

# Selection du graphique à afficher
st.write("   ")
graph_choisi = st.selectbox(label="Sélectionner les variables à étudier", 
                            options=["Age en fonction de Deposit",
                                     "Job en fonction de Deposit",
                                     "Marital en fonction de Deposit",
                                     "Education en fonction de Deposit"],
                            index=None, 
                            placeholder=". . .")

#---------------------------------------
# Age 
if graph_choisi == 'Age en fonction de Deposit':
    
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
elif graph_choisi == 'Job en fonction de Deposit':

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
elif graph_choisi == 'Marital en fonction de Deposit' :
    
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
elif graph_choisi == 'Education en fonction de Deposit' :
    
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
    
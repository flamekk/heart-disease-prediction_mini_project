import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Prédiction Maladie Cardiaque", layout="centered")

# [cite_start]11.1 Charger le modèle sauvegardé [cite: 61]
try:
    model = joblib.load('Model.pkl')
except FileNotFoundError:
    st.error("Le fichier 'Model.pkl' est introuvable. Veuillez exécuter le script d'analyse d'abord.")
    st.stop()

st.title("❤️ Prédiction du Risque Cardiaque (CHD)")
st.write("Cette application utilise un modèle de Machine Learning pour prédire le risque de maladie cardiaque.")

# [cite_start]11.2 Interface de saisie [cite: 62]
st.sidebar.header("Paramètres Cliniques")

def user_input_features():
    # Saisie des variables numériques
    sbp = st.sidebar.number_input('Pression Sanguine (sbp)', min_value=90, max_value=250, value=130)
    tobacco = st.sidebar.number_input('Tabac (kg cumulé)', min_value=0.0, max_value=50.0, value=1.0)
    ldl = st.sidebar.number_input('LDL Cholestérol', min_value=0.0, max_value=20.0, value=4.0)
    adiposity = st.sidebar.number_input('Adiposité', min_value=5.0, max_value=60.0, value=25.0)
    
    # Saisie de la variable catégorielle 'famhist'
    famhist_display = st.sidebar.selectbox('Antécédents familiaux (famhist)', ('Présent', 'Absent'))
    famhist_value = 'Present' if famhist_display == 'Présent' else 'Absent'
    
    typea = st.sidebar.number_input('Comportement Type A', min_value=0, max_value=100, value=50)
    obesity = st.sidebar.number_input('Obésité', min_value=10.0, max_value=60.0, value=25.0)
    alcohol = st.sidebar.number_input('Consommation Alcool', min_value=0.0, max_value=200.0, value=10.0)
    age = st.sidebar.number_input('Age', min_value=15, max_value=90, value=45)

    # [cite_start]11.3 Construire dynamiquement l'exemple [cite: 63]
    data = {
        'sbp': sbp,
        'tobacco': tobacco,
        'ldl': ldl,
        'adiposity': adiposity,
        'famhist': famhist_value,
        'typea': typea,
        'obesity': obesity,
        'alcohol': alcohol,
        'age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Afficher le résumé des saisies
st.subheader('Données patient saisies :')
st.dataframe(input_df)

# Bouton de prédiction
if st.button('Lancer le diagnostic'):
    # [cite_start]11.4 Afficher la prédiction et la probabilité [cite: 65]
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("**Résultat : RISQUE ÉLEVÉ**")
            st.image("https://img.icons8.com/color/96/high-priority.png", width=50)
        else:
            st.success("**Résultat : RISQUE FAIBLE**")
            st.image("https://img.icons8.com/color/96/ok--v1.png", width=50)
            
    with col2:
        st.write("### Probabilités :")
        st.write(f"🟢 Sain : **{proba[0][0]*100:.2f}%**")
        st.write(f"🔴 Malade : **{proba[0][1]*100:.2f}%**")

    # Disclaimer
    st.info("Note : Ce résultat est une estimation statistique et ne remplace pas un avis médical.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Charger le modèle pré-entraîné et le scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialisation de l'historique des prédictions
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Fonction pour normaliser les données
def normalize_data(df, scaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

# Page de description du dataset
def page_description():
    st.title("Description du Dataset")
    st.write("""
    ### Objectif du Modèle
    L'objectif de ce modèle est de prédire les primes d'assurance santé en fonction de divers facteurs tels que l'âge, le sexe, l'IMC, le nombre d'enfants, la région et si l'assuré est fumeur ou non.

    ### Variables du Dataset
    - **Âge**: L'âge de l'assuré.
    - **Sexe**: Le sexe de l'assuré (male/female).
    - **IMC**: L'indice de masse corporelle (IMC) de l'assuré.
    - **Nombre d'enfants**: Le nombre d'enfants à charge.
    - **Région**: La région où vit l'assuré (northeast, northwest, southeast, southwest).
    - **Fumeur**: Si l'assuré est fumeur ou non.
    """)

# Page de prédiction
def page_prediction():
    st.title("Prédiction des Primes d'Assurance")

    # Entrées utilisateur
    age = st.number_input("Âge", min_value=0, max_value=100, value=25)
    sex = st.selectbox("Sexe", ["male", "female"])
    bmi = st.number_input("IMC", min_value=0.0, max_value=100.0, value=25.0)
    region = st.selectbox("Région", ["northeast", "northwest", "southeast", "southwest"])
    smoker = st.selectbox("Fumeur", ["yes", "no"])
    children = st.number_input("Nombre d'enfants", min_value=0, max_value=20, value=0)

    # Création d'un dataframe pour les entrées
    data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'region': [region],
        'smoker': [smoker],
        'children': [children]
    }
    df = pd.DataFrame(data)

    # Encodage des variables catégorielles
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region'])

    # Ajout des colonnes manquantes avec des zéros
    for col in ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']:
        if col not in df.columns:
            df[col] = 0

    # Réordonnancer les colonnes pour correspondre à celles utilisées lors de l'entraînement
    df = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]

    st.write("Données avant normalisation :", df)

    # Bouton de prédiction
    if st.button("Prédire"):
        # Normalisation des variables quantitatives
        df_normalized = normalize_data(df, scaler)
        st.write("Données après normalisation :", df_normalized)

        # Prédiction
        prediction = model.predict(df_normalized)
        st.subheader("Prédiction de la Prime d'Assurance")
        st.write(f"La prime d'assurance estimée est de : ${prediction[0]:.2f}")

        # Ajouter la prédiction à l'historique
        st.session_state['history'].append({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'region': region,
            'smoker': smoker,
            'children': children,
            'prediction': prediction[0]
        })

# Page historique des prédictions
def page_historique():
    st.title("Historique des Prédictions")

    if st.session_state['history']:
        df_history = pd.DataFrame(st.session_state['history'])
        st.write(df_history)
    else:
        st.write("Aucune prédiction effectuée pour le moment.")

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Description", "Prédiction", "Historique"])

# Affichage de la page sélectionnée
if page == "Description":
    page_description()
elif page == "Prédiction":
    page_prediction()
else:
    page_historique()

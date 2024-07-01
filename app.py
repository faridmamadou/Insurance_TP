import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Charger le modèle pré-entraîné (assurez-vous que le modèle est dans le même répertoire ou spécifiez le chemin complet)
model = joblib.load('best_model.pkl')

# Fonction pour normaliser les variables quantitatives
def normalize_data(df, scaler):
    df[['age', 'bmi', 'children']] = scaler.transform(df[['age', 'bmi', 'children']])
    return df

# Chargement du scaler utilisé pendant l'entraînement
scaler = joblib.load('scaler.pkl')

# Interface utilisateur Streamlit
st.title("Prédiction des Primes d'Assurance")

# Entrées utilisateur
age = st.number_input("Âge", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sexe", ["male", "female"])
bmi = st.number_input("IMC", min_value=0.0, max_value=100.0, value=25.0)
region = st.selectbox("Région", ["nord-est", "nord-ouest", "sud-est", "sud-ouest"])
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
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Normalisation des variables quantitatives
df_normalized = normalize_data(df, scaler)

# Prédiction
prediction = model.predict(df_normalized)

st.subheader("Prédiction de la Prime d'Assurance")
st.write(f"La prime d'assurance estimée est de : ${prediction[0]:.2f}")

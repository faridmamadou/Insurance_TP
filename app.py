import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Charger le modèle pré-entraîné (assurez-vous que le modèle est dans le même répertoire ou spécifiez le chemin complet)
model = joblib.load('best_model.pkl')

# Fonction pour normaliser les variables quantitatives
def normalize_data(df, scaler):
    df = scaler.transform(df)
    return df

# Chargement du scaler utilisé pendant l'entraînement
#scaler = joblib.load('scaler.pkl')
scaler = StandardScaler()
# Interface utilisateur Streamlit
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
    'children': [children],
    'smoker': [smoker],
    'region': [region],
    
   
}
df = pd.DataFrame(data)

# Encodage des variables catégorielles
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], prefix='region')
df = df.astype('int')
print(df.head())

# Normalisation des variables quantitatives
df_normalized = pd.DataFrame(scaler.fit_transform(df))

# Prédiction
#prediction = model.predict(df_normalized)

st.subheader("Prédiction de la Prime d'Assurance")
#st.write(f"La prime d'assurance estimée est de : ${prediction[0]:.2f}")

# Importing the necessary libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
loaded_classifier = pickle.load(open('trained_log_model.pkl', 'rb'))
import joblib

st.write("""
# Predicting a Penguin's Sex using Logistic Regression
""")
st.write('---')
X = pd.read_csv('penguins_cleaned.csv')

def user_input_features():
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', float(X['bill_length_mm'].min()), float(X['bill_length_mm'].max()), float(X['bill_length_mm'].mean()))
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', float(X['bill_depth_mm'].min()), float(X['bill_depth_mm'].max()), float(X['bill_depth_mm'].mean()))
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', float(X['flipper_length_mm'].min()), float(X['flipper_length_mm'].max()), float(X['flipper_length_mm'].mean()))
    body_mass_g = st.sidebar.slider('Body Mass (g)', float(X['body_mass_g'].min()), float(X['body_mass_g'].max()), float(X['body_mass_g'].mean()))

    # Categorical feature selection using radio buttons
    species = st.sidebar.radio('Species', ['Adelie', 'Chinstrap', 'Gentoo'])
    island = st.sidebar.radio('Island', ['Biscoe', 'Dream', 'Torgersen'])

    # Encoding categorical variables based on dummy encoding
    species_Chinstrap = 1 if species == 'Chinstrap' else 0
    species_Gentoo = 1 if species == 'Gentoo' else 0
    island_Dream = 1 if island == 'Dream' else 0
    island_Torgersen = 1 if island == 'Torgersen' else 0

    # Creating the user input dataframe
    data = {
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'species_Chinstrap': species_Chinstrap,
        'species_Gentoo': species_Gentoo,
        'island_Dream': island_Dream,
        'island_Torgersen': island_Torgersen
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Collect user input
df = user_input_features()
st.header('Specified Input Parameters')
st.write(df)

# Scale user input
scaler = joblib.load('logistic_regression_scaler.pkl')
df_scaled = scaler.transform(df.values.reshape(1, -1))

# Predict using the model
prediction = loaded_classifier.predict(df_scaled)
def transform_prediction(prediction):
    if prediction == 1:
        adjusted_prediction = 'Male'
        return adjusted_prediction
    else:
        adjusted_prediction = 'Female'
        return adjusted_prediction

# Display prediction
st.header("Prediction of Penguin's Sex")
st.write(f"Predicted Sex: {transform_prediction(prediction)}")
st.write('---')

# Evaluate the model

#Need to use confusion matrix and/or accuracy
from utils import db_connect
engine = db_connect()

# Correct import for pickle
import pickle
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Your Florida Home")
st.subheader("A web app for price analysis")

model = None

# Debug information
print("Current working directory:", os.getcwd())
print("Does EDA folder exist?", os.path.exists("../EDA"))
print("Parent directory files:", os.listdir(".."))


model_path = "models/ligthGBM_algorithm_regressor_default_42.sav"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"🚨 Model not found at path: {model_path}")
    st.stop()
        
print("Model loaded:", type(model))






   

# Fixed dictionary formatting
class_dict = {
    "0": "ZipCode",
    "1": "Bedrooms",
    "2": "Bathrooms",
    "3": "LotSize",
    "4": "YearBuilt",
    "5": "Cooling",
    "6": "Heating",
    "7": "Garage",
    "8": "Property",
    "9": "City",
    "10": "County",
    "11": "Pool"
    
}

 
       
   
property_types = ['apartment','single family', 'townhouse']
cities = ['aventura', 'boca raton','fort lauderdale', 'miami', 'miami beach', 'miami gardens','north miami', 'north miami beach', 'orlando', 'winter garden']

counties = ['broward', 'miami-dade', 'orange', 'palm beach']

val1 = st.selectbox("ZipCode", [i for i in range(33101, 34997)]) # Zipcode range for Florida
val2 = st.selectbox("Bedrooms", [i for i in range(1, 6)])  # Bedrooms options from 1 to 5
val3 = st.selectbox("Bathrooms", [i for i in range(1, 6)])  # Bathrooms options from 1 to 5
val4 = st.selectbox("LotSize", [i * 100 for i in range(1, 11)])  # Square footage from 100 to 1000
val5 = st.selectbox("YearBuilt", [i for i in range(1950, 2024)])
val6 = st.selectbox("Cooling", ["Yes", "No"])  # Cooling options (Yes or No)
val7 = st.selectbox("Heating", ["Yes", "No"])  # Heating options (Yes or No)
val8 = st.selectbox("Garage", ["Yes", "No"])  # Garage options (Yes or No)
val9 = st.selectbox("Choose Property", property_types)
property_vector = [int(pt == val9) for pt in property_types] 
val10 = st.selectbox("Choose City", cities)
city_vector = [int(city == val10) for city in cities]
val11 = st.selectbox("Choose County", counties) 
county_vector = [int(county == val11) for county in counties]
val12 = st.selectbox("Pool", ["Yes", "No"])  # Cooling options (Yes or No)

val6 = 1 if val6 == "Yes" else 0
val7 = 1 if val7 == "Yes" else 0
val8 = 1 if val8 == "Yes" else 0
val12 = 1 if val12 == "Yes" else 0


if st.button("Predict"):
    input_features = [
        int(val1), float(val2), float(val3), float(val4), float(val5),
        int(val6), int(val7), int(val8),
        *property_vector,
        *city_vector,
        *county_vector,
        float(val12)
    ]

    # Predict and format to 2 decimal places
    prediction = round(model.predict([input_features])[0], 2)
    st.write(f"Prediction: ${prediction:,.2f}")


 
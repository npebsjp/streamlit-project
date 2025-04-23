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
           model = pickle.load(open(model_path, "rb"))
        





cities_df = pd.read_csv('./cities.csv')  # Load your CSV file

# Process city names to create a mapping between user-friendly names and model format names
cities_df['cities'] = cities_df['cities'].apply(lambda x: x.replace('city_', ''))

# Create a mapping between the user-friendly city names and the formatted city names
city_mapping = dict(zip(cities_df['cities'], cities_df['cities']))




counties_df = pd.read_csv("counties.csv")


counties_df['counties'] = counties_df['counties'].apply(lambda x: x.replace('county_', ''))

county_mapping = dict(zip(counties_df['counties'], counties_df['counties']))

with open('../EDA/city_columns.pkl', 'rb') as file:
    columns = pickle.load(file)

   

# Fixed dictionary formatting
class_dict = {
    "0": "ZipCode",
    "1": "bedrooms",
    "2": "bathrooms",
    "3": "squareFootage",
    "4": "yearBuilt",
    "5": "cooling",
    "6": "heating",
    "7": "garage",
    "8": "city",
    "9": "county",
    "10": "pool",
    "11": "yearBuilt"
    
}





val1 = st.number_input("ZipCode", min_value=33101, max_value=34997, step=1)  # Zipcode range for Florida
val2 = st.selectbox("Bedrooms", [i for i in range(1, 6)])  # Bedrooms options from 1 to 5
val3 = st.selectbox("Bathrooms", [i for i in range(1, 6)])  # Bathrooms options from 1 to 5
val4 = st.selectbox("Square Footage", [i * 100 for i in range(1, 11)])  # Square footage from 100 to 1000
val5 = st.selectbox("Cooling", ["Yes", "No"])  # Cooling options (Yes or No)
val6 = st.selectbox("Heating", ["Yes", "No"])  # Heating options (Yes or No)
val7 = st.selectbox("Garage", ["Yes", "No"])  # Garage options (Yes or No)


val8 = st.selectbox("Select City", cities_df['cities'], key="city_selectbox_1")
formatted_city = city_mapping[val8]

val9 = st.selectbox("Select County", counties_df['counties'], key="county_selectbox_2")
selected_county = county_mapping[val9]


val10 = st.selectbox("Pool", ["Yes", "No"])  # Pool options (Yes or No)
years = [str(year) for year in range(1950, 2025)]
val10 = st.selectbox("Year Built", years)
val5 = 1 if val5 == "Yes" else 0  # Cooling
val6 = 1 if val6 == "Yes" else 0  # Heating
val7 = 1 if val7 == "Yes" else 0  # Garage
val10 = 1 if val10 == "Yes" else 0  # Pool

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
 


cities_df['cities'] = cities_df['cities'].apply(lambda x: 'city_' + x) 

# Create a mapping from user-friendly city names (e.g., "Alachua") to the model format (e.g., "city_Alachua")
#city_mapping = dict(zip(cities_df['cities'].apply(lambda x: x.replace('city_', '')), cities_df['cities']))
#
## Step 2: Display city selectbox in Streamlit (user-friendly names without 'city_' prefix)
#selected_city_display = st.selectbox("Select City", cities_df['cities'].apply(lambda x: x.replace('city_', '')), key="city_selectbox")
#
## Map the selected city back to the format required by the model (e.g., "city_Alachua")
#formatted_city = city_mapping[selected_city_display]
#
## Step 3: Process counties (similar to cities if needed)
#counties_df['counties'] = counties_df['counties'].apply(lambda x: x.replace('county_', ''))
#
## Step 4: Display county selectbox
#selected_county_display = st.selectbox("Select County", counties_df['counties'], key="county_selectbox")
#
## Map the selected county back to the format required by the model (county_name)
#formatted_county = selected_county_display  # Assuming county name is already in the required format
#
#val_pool = st.selectbox("Pool", ["Yes", "No"], key="pool_selectbox")
#val_pool = 1 if val_pool == "Yes" else 0
#
#years = [str(year) for year in range(1950, 2025)]
#val_year = st.selectbox("Year Built", years, key="year_selectbox")
#
## Step 6: Initialize the cooling, heating, garage, etc.
#val5 = st.selectbox("Cooling", ["Yes", "No"], key="cooling_selectbox")
#val5 = 1 if val5 == "Yes" else 0  # Cooling (1 or 0)
#
#val6 = st.selectbox("Heating", ["Yes", "No"], key="heating_selectbox")
#val6 = 1 if val6 == "Yes" else 0  # Heating (1 or 0)
#
#val7 = st.selectbox("Garage", ["Yes", "No"], key="garage_selectbox")
#val7 = 1 if val7 == "Yes" else 0  # Garage (1 or 0)
#
#city_columns = cities_df['cities'].tolist()
#city_one_hot = [1 if city == formatted_city else 0 for city in city_columns]
#
## One-hot encode the county selection (create a column for each county)
#county_columns = counties_df['counties'].tolist()
#county_one_hot = [1 if county == formatted_county else 0 for county in county_columns]
#
## Step 8: Prepare the input features for the model
## Combine all the features (cooling, heating, etc.) with the one-hot encoded city and county
#input_features = [val5, val6, val7, val_pool, val_year] + city_one_hot + county_one_hot
#
## Step 9: Ensure all input features are numeric
## Convert the input features to float (for any numerical columns, like year, pool, etc.)
#input_features = [float(feature) for feature in input_features]
#

prediction = model.predict([input_features])[0]
#
## Step 11: Display the prediction result
st.write(f"The predicted value for the selected city and county is: {prediction}")
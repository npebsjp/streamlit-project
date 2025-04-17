from utils import db_connect
engine = db_connect()

# Correct import for pickle
import pickle
import streamlit as st
import os

# Change relative path to be more deployment-friendly
# Consider using an absolute path or placing the model in the same directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "EDA", "ligthGBM_algorithm_regressor_default_42.sav")
model = pickle.load(open(model_path, "rb"))

model = pickle.load(open("../EDA/ligthGBM_algorithm_regressor_default_42.sav", "rb"))

# Fixed dictionary formatting
class_dict = {
    "0": "ZipCode",
    "1": "bedrooms",
    "2": "bathrooms",
    "3": "squareFootage"   
}

st.title("Property Prediction")

val1 = st.slider("zipCode", min_value=0.0, max_value=4.0, step=0.1)
val2 = st.slider("bedrooms", min_value=0.0, max_value=4.0, step=0.1)
val3 = st.slider("bathrooms", min_value=0.0, max_value=4.0, step=0.1)
val4 = st.slider("squareFootage", min_value=0.0, max_value=4.0, step=0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
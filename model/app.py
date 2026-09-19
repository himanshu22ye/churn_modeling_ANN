import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

BASE_DIR = Path(__file__).resolve().parent

model=tf.keras.models.load_model(BASE_DIR/'model.keras')

with open(BASE_DIR/'le_gen.pkl','rb') as file:
    le_gen=pickle.load(file)

with open(BASE_DIR/'oneHot_geo.pkl','rb') as file:
    oneHot_geo=pickle.load(file)

#scaled file
with open(BASE_DIR/'scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit App

st.title('Customer Churn Prediction')

#user input
geography=st.selectbox('Geography',oneHot_geo.categories_[0])
gender=st.selectbox('Gender',le_gen.classes_)
age=st.slider('Age',18,90)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one hot geography
geo_encoded=oneHot_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=oneHot_geo.get_feature_names_out(['Geography']))

#combine onehot column with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input data
input_data_scale=scaler.transform(input_data)

#prediction churn
prediction=model.predict(input_data_scale)
prediction_proba=prediction[0][0]
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('GC_churn.joblib')
st.title("Banking Customer Churn Prediction")
st.write("This model predicts if a customer will churn.")

Age = st.number_input("Customer's age", min_value=18)
EstimatedSalary = st.number_input("Customer's estimated salary", min_value=1)
CreditScore = st.number_input("Customer's credit score", min_value=300)
Balance = st.number_input("Customer's bank account balance")
NumOfProducts = st.number_input("Number of accounts open", min_value=1)

prediction = model.predict([[Age, EstimatedSalary, CreditScore, Balance, NumOfProducts]])

if prediction == 0:
    prediction = 'No, this model predicts the customer will not churn.'
else: 
    prediction = 'Yes, this model predicts the customer will churn.'
st.write('Is this customer predicted to churn? {}'.format(prediction))

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the heart disease dataset
heart_data = pd.read_csv('C:\\Users\\adobr\\PycharmProjects\\pythonProject1\\heart_disease_data.csv')

# Split the dataset into features (X) and target variable (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function to predict heart disease
def predict_heart_disease(input_text):
    # Parse input_text to extract individual values
    input_values = list(map(float, input_text.split(',')))
    input_data = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App
st.title('Heart Disease Prediction App')

# Sidebar with a multiline text area
st.sidebar.header('Input Features')

# Get dynamic user input
input_text = st.sidebar.text_area('Input Features (comma-separated)', '43,1,0,120,177,0,0,120,1,2.5,1,0,3')

# Display prediction dynamically based on dynamic user input
if st.sidebar.button('Predict'):
    prediction = predict_heart_disease(input_text)
    st.write('Prediction:', 'The Person has Heart Disease' if prediction else 'The Person does not have a Heart Disease')

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:34 2024

@author: Admin
"""

'Pregnancies' :2,
'Glucose' : 100,
'BloodPressure' : 100,
'SkinThickness' : 10,
'Insulin' : 100,
'BMI' : 25,
'DiabetesPedigreeFunction' : 0.253,
'Age' : 25


'Pregnancies' :1,
'Glucose' : 103,
'BloodPressure' : 30,
'SkinThickness' : 38,
'Insulin' : 83,
'BMI' : 43.3,
'DiabetesPedigreeFunction' : 0.183,
'Age' : 33




# yes

'Pregnancies' :11,
'Glucose' : 143,
'BloodPressure' : 94,
'SkinThickness' : 33,
'Insulin' : 146,
'BMI' : 36.6,
'DiabetesPedigreeFunction' : 0.254,
'Age' : 51


'Pregnancies' :7,
'Glucose' : 160,
'BloodPressure' : 54,
'SkinThickness' : 32,
'Insulin' : 175,
'BMI' : 30.5,
'DiabetesPedigreeFunction' : 0.588,
'Age' : 39


# no

'Pregnancies' :2,
'Glucose' : 107,
'BloodPressure' : 74,
'SkinThickness' : 30,
'Insulin' : 100,
'BMI' : 33.6,
'DiabetesPedigreeFunction' : 0.404,
'Age' : 23

'Pregnancies' :4,
'Glucose' : 151,
'BloodPressure' : 90,
'SkinThickness' : 38,
'Insulin' : 0,
'BMI' : 29.7,
'DiabetesPedigreeFunction' : 0.294,
'Age' : 36

'Pregnancies' :4,
'Glucose' : 141,
'BloodPressure' : 74,
'SkinThickness' : 0,
'Insulin' : 0,
'BMI' : 27.6,
'DiabetesPedigreeFunction' : 0.244,
'Age' : 40

'Pregnancies' :9,
'Glucose' : 106,
'BloodPressure' : 52,
'SkinThickness' : 0,
'Insulin' : 0,
'BMI' : 31.2,
'DiabetesPedigreeFunction' : 0.38,
'Age' : 42



# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler


app = FastAPI()

class model_input(BaseModel):
         
    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int
       
       
# loading the saved model

diabetes_model = pickle.load(open("E:\Ambar\ML Model As API\Diabetic Model Python code\diabetes_model.pkl", 'rb'))

scaler = StandardScaler()


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    input_list = [preg,glu,bp,skin,insulin,bmi,dpf,age]
    
    input_data_as_numpy_array = np.asarray(input_list)
    
    #Reshape the array as we are predicting for one instance
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    std_data = scaler.fit_transform(input_data_reshaped)
    
    prediction = diabetes_model.predict(std_data)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import json

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

diabetes_model = pickle.load(open("E:\Ambar\ML Model As API\Diabetic Model Python code\diabetes_model1.pkl", 'rb'))

scaler = pickle.load(open("E:\Ambar\ML Model As API\Diabetic Model Python code\scaler_object.pkl", 'rb'))


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
    
    
    std_data = scaler.transform(input_data_reshaped)
    
    prediction = diabetes_model.predict(std_data)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
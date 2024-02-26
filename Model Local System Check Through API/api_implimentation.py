# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:00:34 2024

@author: Admin
"""

import json
import requests

#local url:
#url ="http://127.0.0.1:8000/diabetes_prediction"

#public url:
url="https://8238-34-168-136-242.ngrok-free.app/diabetes_prediction"

input_data_for_model={
 'Pregnancies' :4,
 'Glucose' : 141,
 'BloodPressure' : 74,
 'SkinThickness' : 0,
 'Insulin' : 0,
 'BMI' : 27.6,
 'DiabetesPedigreeFunction' : 0.244,
 'Age' : 40

    }

input_jason = json.dumps(input_data_for_model)

response = requests.post(url, data=input_jason)

print(response.text)

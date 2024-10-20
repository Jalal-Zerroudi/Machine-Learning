# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:58:12 2024
Titanic Survival Prediction using Support Vector Machine
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  
import os

def clain_datat(data):
    data['Cabin'] = data['Cabin'].fillna('Unknown')
    
    cabin_counts = data['Cabin'].value_counts()
    
    cabin_mapping = {cabin: idx + 1 for idx, cabin in enumerate(cabin_counts.index)}
    cabin_mapping['Unknown'] = 0
    
    data['CabinNumber'] = data['Cabin'].map(cabin_mapping)
    
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    return data

def get_data(path : str):
    if not os.path.exists(path):
        print(f"Path: {path} does not exist.")
        os._exit(-1)
    
    data = pd.read_csv(path)
    
    data = clain_datat(data)
    
    features = ["Age", "Pclass", "Sex", "SibSp", "Parch", "CabinNumber"]
    X = data[features].copy()
    y = data["Survived"]
    
    return {"features": features, "X": X, "Y": y}

def cerate_Model_Titainc_svm(data , name = "svm_titanic_model"):
    X = data['X']
    y = data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    svm_model = SVC(kernel='linear')
    
    svm_model.fit(X_train, y_train)
    
    joblib.dump(svm_model, name+'.pkl')
    print(f"Model saved as '{name}.pkl'")
    
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

def load_model_and_predict(X_new):
    svm_model = joblib.load('svm_titanic_model.pkl')
    print("Model loaded successfully.")
    
    predictions = svm_model.predict(X_new)
    return predictions

def main() -> None:
    path : str = "D:\\002 - Jalal && Python\\Titanic\\train.csv"
    data = get_data(path)
    cerate_Model_Titainc_svm(data,name="svm_titanic_model")

if __name__ == '__main__':
    main()

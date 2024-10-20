# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:57:54 2024
@author: Mon Pc
"""

import All_Jalal_Model.model_titanic_svm as ms
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np

def Train_model() -> None:
    """Trains the SVM model on the Titanic dataset."""
    path = "./train.csv" 
    if not os.path.exists(path):
        messagebox.showerror("Error", f"Path: {path} does not exist.")
        return

    data = ms.get_data(path)
    ms.create_model_titanic_svm(data, name="svm_titanic_model")
    messagebox.showinfo("Success", "Model trained successfully!")

def test_Model() -> None:
    """Loads the trained model and makes predictions on new data."""
    
    path_test = "./test.csv"
    if not os.path.exists(path_test):
        print(f"Path: {path_test} does not exist.")
        return

    data = pd.read_csv(path_test)
    
    data = ms.clean_data(data)  
    
    features = ["Age", "Pclass", "Sex", "SibSp", "Parch", "CabinNumber"]
    
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        print(f"Missing features in test data: {', '.join(missing_features)}")
        return

    X_new = data[features].copy()
    
    predictions = ms.load_model_and_predict(X_new)
    
    plot_predictions(predictions)

def plot_predictions(predictions):
    """Plots a bar chart of the prediction distribution."""

    unique, counts = np.unique(predictions, return_counts=True)
    prediction_counts = dict(zip(unique, counts))
    
    labels = ['Did Not Survive', 'Survived']
    values = [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['red', 'green'])
    plt.xlabel('Survival Status')
    plt.ylabel('Number of Passengers')
    plt.title('Predictions for Titanic Survival')
    plt.grid(axis='y')

    plt.show()

def create_gui():

    window = tk.Tk()
    window.title("Titanic Model Trainer & Tester")
    window.geometry("400x200")


    label = tk.Label(window, text="Select an option:", font=("Arial", 14))
    label.pack(pady=20)


    train_button = tk.Button(window, text="Train Model", command=Train_model, width=20, font=("Arial", 12))
    train_button.pack(pady=10)

    test_button = tk.Button(window, text="Test Model", command=test_Model, width=20, font=("Arial", 12))
    test_button.pack(pady=10)


    window.mainloop()

if __name__ == '__main__':
    create_gui()

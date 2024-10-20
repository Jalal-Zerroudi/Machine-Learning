# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:57:54 2024

@author: Mon Pc
"""

import All_Jalal_Model.model_titanic_svm as ms
import pandas as pd

def main():
    path : str = "D:\\002 - Jalal && Python\\Titanic\\test.csv"
    
    data = pd.read_csv(path)

    data = ms.clain_datat(data)
    
    features = ["Age", "Pclass", "Sex", "SibSp", "Parch", "CabinNumber"]
    
    X = data[features].copy()

    print(X)
    
    predictions = ms.load_model_and_predict(X)
    
    data['Survived'] = predictions
    
    result = data[['PassengerId', 'Survived']]
    result.to_csv(r"D:\002 - Jalal && Python\Titanic\predictions.csv", index=False)
    
    print(result)

    print("Predictions saved as 'predictions.csv'")
    
if __name__ == '__main__':
    main()

"""Главный python модуль"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sgd import SGD

LAST_ATTR = "MEDV"

file_path = os.path.abspath("./resources/boston.csv")
data = pd.read_csv(file_path)
X = data.drop(LAST_ATTR, axis=1).values
y = data[LAST_ATTR].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

sgd = SGD()

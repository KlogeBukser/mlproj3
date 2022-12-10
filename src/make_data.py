# make_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_wine_data():
    wine_data = pd.read_csv("Dataset/winequality-red.csv")
    # print(wine_data.columns)

    x = wine_data[wine_data.keys().drop("quality")]
    y = wine_data["quality"]

    return train_test_split(x, y, train_size=0.8, test_size=0.2)

x_train, x_test, y_train, y_test = read_wine_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# make_data.py
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split

def read_wine_data():
    wine_data = pd.read_csv("Dataset/winequality-red.csv")
    wine_data = pd.DataFrame(wine_data)
    # print(wine_data.columns)

    x = wine_data[wine_data.keys().drop("quality")].values
    y = wine_data["quality"].values

    return train_test_split(x, y, train_size=0.8, test_size=0.2)

def visualise_wine_data():
    return

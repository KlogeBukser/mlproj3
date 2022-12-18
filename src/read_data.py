# make_data.py
import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plot_tools import *

def read_wine_data(is_visual=False):
    '''returns X_train, X_test, y_train, y_test from wine data, 
    not turn into one hot vector yet'''
    wine_data = pd.read_csv("Dataset/winequality-red.csv")
    wine_data = pd.DataFrame(wine_data)
    # print(wine_data.columns)

    x = wine_data[wine_data.keys().drop("quality")].values
    y = wine_data["quality"].values

    if is_visual:
        visualise_wine(wine_data)

    return train_test_split(x, y, train_size=0.8, test_size=0.2)

def visualise_wine(df):
    sns.set()
    df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    save_fig("data-visualisation.pdf")
    heatmap(df.corr(), title="Feature Correlation Matrix", filename="feature-corr.pdf")


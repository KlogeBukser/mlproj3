# make_data.py
import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plot_tools import *


def read_wine_data(is_visual=False, split=True):
    '''returns X_train, X_test, y_train, y_test from wine data, 
    not turn into one hot vector yet'''
    wine_data = pd.read_csv("Dataset/winequality-red.csv")
    wine_data = pd.DataFrame(wine_data)
    # print(wine_data.columns)

    x = wine_data[wine_data.keys().drop("quality")].values
    y = wine_data["quality"].values

    if is_visual:
        visualise_wine(wine_data)

    if split:
        return train_test_split(x, y, train_size=0.8, test_size=0.2)
    return x,y

def visualise_wine(df):
    sns.set()
    df.hist(figsize=(7, 8), bins=30, xlabelsize=12, ylabelsize=12)
    plt.tight_layout()
    save_fig("data-visualisation.pdf")
    sns.heatmap(df.corr(),cmap='Blues')
    set_paras('','',title="Feature Correlation Matrix", filename="feature-corr.pdf")


if __name__ == "__main__":
    read_wine_data(True)
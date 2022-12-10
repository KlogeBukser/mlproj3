# tree.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from read_data import *
from plot_tools import *
from NN.misc import accuracy_score

def gini(p):
    return 2*p*(1-p)


x_train, x_test, y_train, y_test = read_wine_data()
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)

tree_clf.fit(x_test, y_test)
y_pred = tree_clf.predict(x_test)

print(accuracy_score(y_pred, y_test))

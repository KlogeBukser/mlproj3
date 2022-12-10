# tree.py
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from read_data import *
from plot_tools import *
# from NN.misc import accuracy_score

def 
max_depth = int(sys.argv[1])

x_train, x_test, y_train, y_test = read_wine_data()
tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

tree_clf.fit(x_train, y_train)
y_pred = tree_clf.predict(x_test)
print(tree_clf.score(x_test, y_test))



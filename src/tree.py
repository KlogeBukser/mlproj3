# tree.py
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from read_data import *
from plot_tools import *
# from NN.misc import accuracy_score

def run_tree(max_depth=False):

    if not max_depth:
        # print(max_depth)
        max_depth = int(sys.argv[1])

    x_train, x_test, y_train, y_test = read_wine_data()
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    tree_clf.fit(x_train, y_train)
    y_pred = tree_clf.predict(x_test)
    return tree_clf, tree_clf.score(x_test, y_test)

def run_forest(max_depth, max_run):
    best_tree = None
    best_acc = 0
    scores = []
    for _ in range(max_run):
        tree, acc = run_tree(max_depth)
        if acc > best_acc:
            best_tree = tree
            best_acc = acc
        scores.append(acc)
    
    # print(best_acc)
    return scores

def find_best_depth(max_run):
    mean_scores = []
    max_depths = np.arange(5,50,5) 
    for max_depth in max_depths:
        mean_scores.append(np.mean(run_forest(max_depth, max_run))) # run forest run!

    plt.plot(max_depths, mean_scores)
    set_paras("Max Depth","Accuracy Score", "Mean Accuracy Score for " + str(max_run) + " runs vs Max Depth", 
    "tree-accVSdepth.pdf")

find_best_depth(500)

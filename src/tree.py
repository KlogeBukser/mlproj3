# tree.py
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from plot_tools import *

def make_tree(X_train, y_train, max_depth=21):
    
    tree_rgn = DecisionTreeRegressor(max_depth=max_depth)
    tree_rgn.fit(X_train, y_train)
 
    return tree_rgn

def run_tree(X_train, y_train, X_test, y_test, max_depth, max_run, is_rgn):

    ''' is_r2: Bool, only used when is_rgn, determines if run_tree 
        return R2 score or accuracy for comparison to classification '''
    scores =  []
    accs = []
    mses = []
    for _ in range(max_run):
        if is_rgn: 
            tree = make_tree(X_train, y_train, max_depth)
            y_pred = tree.predict(X_test)
            accs.append((y_test == y_pred).sum()/len(y_test))
            mses.append(MSE(y_test, y_pred))
        else:
            tree = make_tree_clf(X_train, y_train, max_depth)
        score = tree.score(X_test, y_test)
        scores.append(score)
    
    if is_rgn:
        return np.mean(scores), np.mean(accs), np.mean(mses)
    return np.mean(scores)

def tune_depth(X_train, y_train, X_test, y_test, max_run, low, high, step, is_rgn=True):
    mean_scores = []
    mean_accs = [] # only used for rgn
    mean_mses = []
    max_depths = np.arange(low,high,step) 
    for max_depth in max_depths:
        if is_rgn:
            score, acc, mse = run_tree(X_train, y_train, X_test, y_test, max_depth, max_run, is_rgn)
            mean_scores.append(score)
            mean_accs.append(acc)
            mean_mses.append(mse)
        else:
            score, mse = run_tree(X_train, y_train, X_test, y_test, max_depth, max_run, is_rgn)
            mean_mses.append(mse)
            mean_scores.append(score)
       
    if is_rgn:
        plot_2D(max_depths, [mean_scores, mean_accs, mean_mses], 3, x_title="Max Depth ["+ str(low) + "," + str(high)+ "]", y_title=r"Score", 
            title=r"Mean Score for " + str(max_run) + " runs vs Max Depth", 
            label=[r"$R^2$", "Accuracy", "MSE"],
            filename="tree-rgn-scoresVSdepth["+ str(low) + "," + str(high)+ "].pdf")

    else:
        plot_2D(max_depths, [mean_scores, mean_mses], 2, x_title="Max Depth ["+ str(low) + "," + str(high)+ "]", y_title=r"Accuracy Score", 
            title=r"Mean Score for " + str(max_run) + " runs vs Max Depth", 
            label=["Accuracy", "MSE"],
            filename="tree-clf-scoresVSdepth["+ str(low) + "," + str(high)+ "].pdf")

def make_tree_clf(X_train, y_train, max_depth):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    tree_clf.fit(X_train, y_train)
    return tree_clf

def make_forest(X_train, X_test, y_train, y_test, max_depth):
    model = RandomForestRegressor(n_estimators=500, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def make_forest_clf(X_train, X_test, y_train, y_test, max_depth):
    model = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def MSE(z_test, z_pred):
    return np.mean((z_test - z_pred)**2)

if __name__ == "__main__":

    from read_data import read_wine_data
    X_train, X_test, y_train, y_test = read_wine_data()
    
    tune_depth(X_train, y_train, X_test, y_test, 1000, 1, 20, 2)
    tune_depth(X_train, y_train, X_test, y_test, 1000, 5, 45, 5, is_rgn=False) 

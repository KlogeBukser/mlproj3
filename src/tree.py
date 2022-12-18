# tree.py
from cProfile import label
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix

from plot_tools import *

def make_tree(X_train, y_train, max_depth=21):
    
    tree_rgn = DecisionTreeRegressor(max_depth=max_depth)
    tree_rgn.fit(X_train, y_train)
 
    return tree_rgn

def run_tree(X_train, y_train, X_test, y_test, max_depth, max_run, is_rgn):

    ''' is_r2: Bool, only used when is_rgn, determines if run_tree 
        return R2 score or accuracy for comparison to classification '''
    r2s =  []
    accs = []
    mses = []
    for _ in range(max_run):
        if is_rgn: 
            tree = make_tree(X_train, y_train, max_depth)
            
        else:
            tree = make_tree_clf(X_train, y_train, max_depth)
            
        y_pred = tree.predict(X_test)
        accs.append(accuracy(y_test, y_pred))
        mses.append(MSE(y_test, y_pred))
        r2 = R2(y_test, y_pred)
        r2s.append(r2)
    
    return np.mean(r2s), np.mean(accs), np.mean(mses)

def tune_depth(X_train, y_train, X_test, y_test, max_run, low, high, step, is_rgn=True):
    mean_r2 = []
    mean_accs = []
    mean_mses = []
    max_depths = np.arange(low,high,step) 
    for max_depth in max_depths:
        r2, acc, mse = run_tree(X_train, y_train, X_test, y_test, max_depth, max_run, is_rgn)
        mean_r2.append(r2)    
        mean_accs.append(acc)
        mean_mses.append(mse)
       
    if is_rgn:
        plot_2D(max_depths, [mean_r2, mean_accs, mean_mses], 3, x_title="Max Depth ["+ str(low) + "," + str(high)+ "]", y_title=r"Score", 
            title=r"Decision Tree (rgn) Mean Score for " + str(max_run) + " runs vs Max Depth", 
            label=[r"$R^2$", "Accuracy", "MSE"],
            filename="tree-rgn-scoresVSdepth["+ str(low) + "," + str(high)+ "].pdf")

    else:
        plot_2D(max_depths, [mean_r2, mean_accs, mean_mses], 3, x_title="Max Depth ["+ str(low) + "," + str(high)+ "]", y_title=r"Accuracy Score", 
            title=r"Decision Tree (clf) Mean Score for " + str(max_run) + " runs vs Max Depth", 
            label=[r"$R^2$","Accuracy", "MSE"],
            filename="tree-clf-scoresVSdepth["+ str(low) + "," + str(high)+ "].pdf")

def make_tree_clf(X_train, y_train, max_depth):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    tree_clf.fit(X_train, y_train)
    return tree_clf

def make_forest(X_train, y_train, max_depth):
    model = RandomForestRegressor(n_estimators=500, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def make_forest_clf(X_train, y_train, max_depth):
    model = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def run_forest(X_train, X_test, y_train, y_test, max_depth=15, max_run=1):
    
    clf_scores = []
    rgn_scores = []
    
    for _ in range(max_run):
        clf = make_forest_clf(X_train, y_train, max_depth=15) 
        rgn = make_forest(X_train, y_train, 15)

        rgn_pred = rgn.predict(X_test)
        rgn_pred_round = rgn_pred.round() 
        rgn_score = accuracy(y_test, rgn_pred_round)

        clf_pred = clf.predict(X_test)
        clf_scores.append(clf.score(X_test, y_test))
        rgn_scores.append(rgn_score)
    
    rgn_cm = confusion_matrix(y_test, rgn_pred_round)
    clf_cm = confusion_matrix(y_test, clf_pred)

    labels = [3,4,5,6,7,8]
    heatmap(rgn_cm, labels, labels, annot = True, title = "Random Forest regression",
    filename="tree-forest-rgncm.pdf")
    heatmap(clf_cm, labels, labels, annot = True, title = "Random Forest classification",
    filename="tree-forest-clfcm.pdf")

    plot_2D(range(max_run), [rgn_scores, clf_scores], 2, 
        title="Random Forest scores for "+str(max_run)+" runs with depth "+str(max_depth),
        x_title="Run", y_title="Accuracies", 
        label=["Regression","Classification"],
        filename="tree-forest-scores"+str(max_depth)+".pdf")
    


def MSE(z_test, z_pred):
    return np.mean((z_test - z_pred)**2)

def accuracy(y_test, y_pred):
    return (y_test == y_pred).sum()/len(y_test)

def R2(z_test,z_pred): 
    return 1 - np.mean((z_test - z_pred)**2)/np.mean((z_test - np.mean(z_test))**2)

if __name__ == "__main__":

    from read_data import read_wine_data
    X_train, X_test, y_train, y_test = read_wine_data()
    
    tune_depth(X_train, y_train, X_test, y_test, 1000, 1, 20, 1)
    tune_depth(X_train, y_train, X_test, y_test, 1000, 1, 45, 5, is_rgn=False) 

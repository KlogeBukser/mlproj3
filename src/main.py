# main.py

from sklearn.metrics import confusion_matrix
from read_data import *
from lin_rgn import *
from nn import *
from tree import *


def run_lin_rgn():

    ols_score, ols_pred = ols(X_train, X_test, y_train, y_test)
    ridge_score, ridge_pred = ridge(X_train, X_test, y_train, y_test, 0.001)
    lasso_score, lasso_pred = lasso(X_train, X_test, y_train, y_test, 0.001)

    o_cm = confusion_matrix(y_test, ols_pred.round())
    r_cm = confusion_matrix(y_test, ridge_pred.round())
    l_cm = confusion_matrix(y_test, lasso_pred.round())

    heatmap(o_cm, title="OLS Confusion Matrix\n"+r"$R^2 = $"+str(round(ols_score, 4)),  
        filename="lin_rgn-ols-cm.pdf",annot=True, xticklabels=labels, yticklabels=labels )
    heatmap(r_cm, title="Ridge Confusion Matrix\n"+r"$R^2 = $"+str(round(ridge_score,4)),  
        filename="lin_rgn-ridge-cm.pdf",annot=True, xticklabels=labels, yticklabels=labels )
    heatmap(l_cm, title="Lasso Confusion Matrix\n"+r"$R^2 = $"+str(round(lasso_score,4)),  
        filename="lin_rgn-lasso-cm.pdf",annot=True, xticklabels=labels, yticklabels=labels )

def run_nn():
    model = nn_rgn(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)

    nn_cm = confusion_matrix(y_test, y_pred.round())

    heatmap(nn_cm, annot=True, xticklabels=labels, yticklabels=labels,
        title="Neural Network Regression Confusion Matrix \n"+r"$R^2 = $"+str(round(score,4)), 
        filename="nn-rgn-cm.pdf",
        )

def cmp_all(max_run=100):

    collector = np.zeros((6,max_run))
    x,y = read_wine_data(split=False)
    for i in range(max_run):
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        ols_pred = ols(X_train, X_test, y_train, y_test)[1]
        ridge_pred = ridge(X_train, X_test, y_train, y_test, 0.001)[1]
        lasso_pred = lasso(X_train, X_test, y_train, y_test, 0.001)[1]
        collector[0][i] = accuracy(y_test, ols_pred.round())
        collector[1][i] = accuracy(y_test, ridge_pred.round()) 
        collector[2][i] = accuracy(y_test, lasso_pred.round()) 

        nn = nn_rgn(X_train, y_train)
        nn_pred = nn.predict(X_test)
        collector[3][i] = accuracy(y_test, nn_pred.round())

        forest_rgn = make_forest(X_train, y_train, 15)
        forest_rgn_pred = forest_rgn.predict(X_test)
        collector[4][i] = accuracy(y_test, forest_rgn_pred.round())

        forest_clf = make_forest_clf(X_train, y_train, 15)
        forest_clf_pred = forest_clf.predict(X_test)
        collector[5][i] = accuracy(y_test, forest_clf_pred)
    

    plot_2D(np.arange(max_run), collector, 6, "All Methods Compared", "Runs", 
            "Accuracy", ['OLS','Ridge', 'Lasso', 'FFNN', 'RF rgn', 'RF clf'], 
            "final-cmp.pdf")

if __name__ == "__main__":

    from read_data import read_wine_data
    labels = [3,4,5,6,7,8]
    X_train, X_test, y_train, y_test = read_wine_data()
    
    run_lin_rgn()
    run_nn()
    max_run = int(sys.argv[1])
    run_forest(X_train, X_test, y_train, y_test, max_run=max_run)
    cmp_all(max_run)




# linear regression
from sklearn import linear_model
import numpy as np
from plot_tools import *


def ols(X_train, X_test, y_train, y_test):
    # print("Linear regression")
    ols = linear_model.LinearRegression()
    ols.fit(X_train, y_train)
    ols_pred = ols.predict(X_test)
    score = ols.score(X_test, y_test)
    # print((y_test == ols_pred.round()).sum()/test_len)

    return score, ols_pred


def ridge(X_train, X_test, y_train, y_test, alpha):
    # print("Ridge")
    # print("alpha", alpha)
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    score = ridge.score(X_test, y_test)
    # print((y_test == ridge_pred.round()).sum()/test_len)

    return score, ridge_pred
    

def lasso(X_train, X_test, y_train, y_test, alpha):
    # print("Lasso")
    # print("alpha", alpha)
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    score = lasso.score(X_test, y_test)
    # print((y_test == lasso_pred.round()).sum()/test_len)
    return score, lasso_pred

def tune_hyperparams(X_train, X_test, y_train, y_test):

    ridge_scores = []
    lasso_scores = []
    params = np.logspace(-4,0,1000)

    for alpha in params:
        ridge_scores.append(ridge(X_train, X_test, y_train, y_test, alpha)[0])
        lasso_scores.append(lasso(X_train, X_test, y_train, y_test, alpha)[0])

    plot_2D(params, [ridge_scores, lasso_scores], plot_count=2, 
        title="Hyperparameter Tuning for Linear Models",  
        x_title=r"$\lambda$", y_title=r"$R^2$ scores", 
        label=["Ridge", "Lasso"], filename="lin_rgn-hyperparams.pdf")

    # print("Ridge and Lasso R2 scores")
    # print("Best Ridge score:", np.max(ridge_scores))
    # print("Best Lasso score:", np.max(lasso_scores))


if __name__ == "__main__":
    from read_data import read_wine_data
    X_train, X_test, y_train, y_test = read_wine_data() 
    tune_hyperparams(X_train, X_test, y_train, y_test)
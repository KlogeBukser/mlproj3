# nn.py

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from plot_tools import heatmap, plot_2D

# the default values are adjusted accoridng to the tuning result
def nn_rgn(X_train, y_train, hidden_size=(100,), activation = "logistic", 
    solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', 
    learning_rate_init=0.001, power_t=0.5, max_iter=500):
    # print("MLPRegressor")
    model = MLPRegressor(random_state=1, hidden_layer_sizes=hidden_size, max_iter=max_iter, 
    activation=activation, learning_rate=learning_rate)
    model.fit(X_train, y_train) 
    return model

def method_tuning(X_train, X_test, y_train, y_test, max_iter):

    scores = np.zeros((4, 3)) 
    activations = ["relu", "tanh", "logistic", "identity"]
    methods = ["constant", "invscaling", "adaptive"]
    for i, activation in enumerate(activations):
        for j, method in enumerate(methods):
            best_score = -999
            for _ in range(1):
                model = nn_rgn(X_train, y_train, 
                activation=activation, learning_rate=method, max_iter=max_iter)
                score = model.score(X_test, y_test)
                if score > best_score:
                    best_score = score
            scores[i][j] = best_score
    
    heatmap(scores, methods, activations,  "Learning Rate Update", "Activation Function",
    True, r"$R^2$ for Different Methods and Activation Functions", 
    filename = "nn-methods-cm("+str(max_iter)+" iter).pdf")

    print("Sucessfully tested methods!")


def hidden_size_tuning(X_train, X_test, y_train, y_test):

    scores = [] 
    for hidden_size in [(100), (100,100), (50,50,50)]:
        model = nn_rgn(X_train, y_train, hidden_size=hidden_size, 
        activation="logistic", solver="lbfgs")
        scores.append(model.score(X_test, y_test))
    
    # print(scores)
    plot_2D(np.arange(1,4), scores, title=r"$R^2$ Score vs Number of Hidden Layers",
    x_title = "Number of Hidden Layers", y_title = r"$R^2$", 
    filename="nn-hidden.pdf")


if __name__ == "__main__":
    from read_data import read_wine_data
    import sys
    X_train, X_test, y_train, y_test = read_wine_data()
    max_iter = sys.argv[1]
    method_tuning(X_train, X_test, y_train, y_test, max_iter)
    hidden_size_tuning(X_train, X_test, y_train, y_test)
    



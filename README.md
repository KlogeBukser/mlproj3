# About
This repository contains the source code, the dataset and report to FYS-STK4155 project 3.
For this project we explore different machine learning methods (both regression and classification) on [the dataset](#dataset).

## Dataset
 using the [red wine quality data](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download) from Kaggle. This dataset can also be found in the under the *Dataset* folder. 

## Structure
All code is contain in the [src](https://github.com/KlogeBukser/mlproj3/tree/main/src) Each machine learning method (Linear Regression, Neural Network and Decision Tree) is written and should be run separetely from its own file and done using **makefile** with instructions [below](#execution).

# Prerequisites
Check you have *make* installed by running:
```sh
make --version
```

Check you have the following dependancies:
- numpy
- matplotlib
- scikit-learn
- seaborn
otherwise run:
```sh
make install
```


# Execution
We've used makefile for selected runs. They are structured so that
- run all functions
```sh
make all
```

- run with linear models (OLS, Ridge, Lasso)
```sh
make lin_rgn $(show_tuning)
```


- run with neural network
```sh
make nn
```

- run with basic decision trees
```sh
make tree
```

- run with random forest
```sh
make forest
```

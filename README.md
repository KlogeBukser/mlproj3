# About
This repository contains the source code, the dataset and report to FYS-STK4155 project 3.
For this project we discussed different methods for classification using the [red wine quality data](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download) from Kaggle. This dataset can also be found in the under the *Dataset* folder.

# Prerequisite
Check you have PyTorch installed by running:
```sh
python3 -c "import torch"
```
at your terminal.
If this throws an error, install PyTorch by simply running:
```sh
pip3 install torch
```
p.s. I've used PyTorch because somehow I've had a lot of trouble installing Tensorflow on my Mac M1 and have subsequently decided to use PyTorch instead. Tensorflow also has no official MacOS GPU support.

Check you have *make* installed by running:
```sh
make --version
```

# Execution
We've used makefile for selected runs. They are structured so that
- run all functions
```sh
make all
```

- run with neural network from project 2
```sh
make nn
```
- run with neural networks from PyTorch (both dense neural network and convoluted neural network)
```sh
make torch
```

- run with basic decision trees
```sh
make tree
```

- run with random forest
```sh
make forest
```

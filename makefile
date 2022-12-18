all: main 

max_depth := 10

lin_rgn:
	python3 src/lin_rgn.py
tree-clf:
	python3 src/tree.py $(max_depth)

nn:
	python3 src/ffnn.py

install:
	pip3 install numpy
	pip3 install scikit-learn
	pip3 install matplotlib
	pip3 install seaborn
clean: # removes all plots
	rm -r -f plots/*.pdf
	rm -r -f plots/*.png


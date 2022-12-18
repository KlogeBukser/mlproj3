all: data lin_rgn nn tree main


max_run:= 100
main:
	python3 src/main.py $(max_run)

data:
	python3 src/read_data.py
lin_rgn:
	python3 src/lin_rgn.py
tree:
	python3 src/tree.py
nn:
	python3 src/nn.py $(max_iter)

install:
	pip3 install numpy
	pip3 install scikit-learn
	pip3 install matplotlib
	pip3 install seaborn
clean: # removes all plots
	rm -r -f plots/*.pdf
	rm -r -f plots/*.png


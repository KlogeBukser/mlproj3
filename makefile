all: tree

max_depth := 10
tree:
	python3 src/tree.py $(max_depth)

clean: # removes all plots
	rm -r -f plots/*.pdf
	rm -r -f plots/*.png


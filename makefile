all: tree

max_depth := 7
tree:
	python3 src/tree.py $(max_depth)

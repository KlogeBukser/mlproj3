# ai.py
import re
import numpy as np

rng = np.random.default_rng()
class Ai:
    def __init__(self) -> None:
        return

    def gen_move(self) -> bool: # if bird should jump
        print(rng.random())
        if rng.random() > 0.5:
            return True
        else:
            return False
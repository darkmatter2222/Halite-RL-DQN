import numpy as np
from itertools import permutations
perm = permutations([1, 2, 3, 4, 5])

x = 0
for i in list(perm):
    print(i)
    x += 1

print(x)

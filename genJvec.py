import numpy as np
from itertools import combinations

def genJvec(N):

    if N <= 1:
        return []
    
    return [int(str(a[0])+str(a[1])) for a in combinations(range(1, N+1), 2)]

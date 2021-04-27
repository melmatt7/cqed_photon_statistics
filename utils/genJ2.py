import numpy as np

def genJ2(jvec):
    nck2=len(jvec)
    N=round(np.sqrt(2*nck2+0.25)+0.5)

    if N<2:
        return []
    elif N==2:
        return 0

    
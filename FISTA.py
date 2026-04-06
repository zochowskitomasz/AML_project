import numpy as np

def FISTA(X, y, lam, bet, iterations=500):
    def ST(B, lam):
        n = len(B)
        S = [0 for i in range(n)]

        for i in range(n):
            if B[i] > lam:
                S[i] = B[i] - lam

            elif B[i] < -lam:
                S[i] = B[i] + lam

        return S
    
    def g(b):
        return 1/2 * (np.linalg.norm(y - X @ b, ord=2)**2)

    def Dg(b):
        return X.T @ (X @ b - y)

    bk = bet
    bkt = bet
    tk = 1
    tkt = 1

    for i in range(2, iterations):
        # get v
        v = bk + ((i-2)/(i+1)) * (bk - bkt)

        #update beta
        bkt = bk

        #update step
        tk = tkt

        while True:
            tk = 0.1 * tk
            bk = ST(v + tk * Dg(v), lam * tk)

            if g(bk) <= g(v) + Dg(v).T @ (bk - v) + 1/(2*tk) * np.linalg.norm(bk - v, ord=2)**2:
                break
        
        tk = tk / 0.1
        bk = ST(v + (tk * X.T) @ (y - X*v), lam * tk)
        tkt = tk


    return bk
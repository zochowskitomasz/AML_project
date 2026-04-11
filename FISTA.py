import numpy as np

def FISTA(X, y, lam, bet, iterations=500):
    def ST(B, lam):
        n = len(B)
        S = np.zeros((n, 1))

        for i in range(n):
            if B[i][0] > lam:
                S[i][0] = B[i][0] - lam

            elif B[i][0] < -lam:
                S[i][0] = B[i][0] + lam

        return S
    
    def g(b):
        return 1/2 * (np.linalg.norm(y - X @ b, ord=2)**2)

    def Dg(b):
        return X.T @ (X @ b - y)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    bk = bet
    bkt = bet
    tk = 1

    for i in range(2, iterations):
        # get v
        v = bk + ((i-2)/(i+1)) * (bk - bkt)

        #update beta
        bkt = bk

        while True:
            tk = 0.9 * tk
            bk = ST(v + tk * Dg(v), lam * tk)

            if g(bk) <= g(v) + Dg(v).T @ (bk - v) + 1/(2*tk) * np.linalg.norm(bk - v, ord=2)**2:
                break
        
        bk = ST(v + (tk * X.T) @ (y - X @ v), lam * tk)

        if tk < 1e-10:
            break


    return bk
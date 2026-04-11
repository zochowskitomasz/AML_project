import numpy as np

def FISTA(X, y, lam, bet, iterations=500, fit_intercept=True):
    def ST(B, lam):
        S = np.zeros_like(B)
        n = len(B)

        for i in range(n):
            if fit_intercept and i == n - 1:
                S[i] = B[i]
            elif B[i] > lam:
                S[i] = B[i] - lam
            elif B[i] < -lam:
                S[i] = B[i] + lam

        return S
    
    def g(b):
        return 1/2 * (np.linalg.norm(y - X @ b, ord=2)**2)

    def Dg(b):
        return - X.T @ (y - X @ b)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    if fit_intercept:
        X = np.column_stack([X, np.ones((X.shape[0], 1))])

    bet = np.array(bet)
    if bet.ndim == 1:
        bet = bet.reshape(-1, 1)
    if fit_intercept and bet.shape[0] == X.shape[1] - 1:
        bet = np.vstack([bet, np.zeros((1, 1))])
    elif bet.shape[0] != X.shape[1]:
        raise ValueError("bet must have one entry per feature, or one fewer when fit_intercept=True.")

    bk = bet
    bkt = bet
    tk = 1

    for i in range(2, iterations):
        # get v
        v = bk + ((i-2)/(i+1)) * (bk - bkt)

        #update beta
        bkt = bk

        while True:
            bk = ST(v - tk * Dg(v), lam * tk)

            if g(bk) <= g(v) + Dg(v).T @ (bk - v) + 1/(2*tk) * np.linalg.norm(bk - v, ord=2)**2:
                break
            else:
                tk = 0.9 * tk

    if fit_intercept:
        return bk[:-1], bk[-1]
    return bk
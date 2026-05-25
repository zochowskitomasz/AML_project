import numpy as np

def FISTA(X, y, lam, bet, iterations=500, fit_intercept=True):
    def sigmoid(values):
        values = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

    def soft_threshold(values, threshold):
        result = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
        if fit_intercept:
            result[-1] = values[-1]
        return result

    def gradient(theta):
        probabilities = sigmoid(X @ theta)
        return (X.T @ (probabilities - y)) / X.shape[0]

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    y = y.astype(float, copy=False)
    unique_values = np.unique(y)
    if unique_values.size != 2:
        raise ValueError("y must contain exactly two classes.")
    if not np.all(np.isin(unique_values, [0.0, 1.0])):
        y = (y == unique_values.max()).astype(float)

    if fit_intercept:
        X = np.column_stack([X, np.ones((X.shape[0], 1))])

    bet = np.array(bet)
    if bet.ndim == 1:
        bet = bet.reshape(-1, 1)
    if fit_intercept and bet.shape[0] == X.shape[1] - 1:
        bet = np.vstack([bet, np.zeros((1, 1))])
    elif bet.shape[0] != X.shape[1]:
        raise ValueError("bet must have one entry per feature, or one fewer when fit_intercept=True.")

    bk = bet.astype(float, copy=True)
    bkt = bk.copy()

    spectral_norm = np.linalg.norm(X, ord=2)
    lipschitz = (spectral_norm * spectral_norm) / (4.0 * X.shape[0])
    step_size = 1.0 / max(lipschitz, 1e-12)

    for i in range(iterations):
        v = bk + ((i - 1) / (i + 2)) * (bk - bkt) if i > 0 else bk
        bkt = bk.copy()

        candidate = v - step_size * gradient(v)
        bk = soft_threshold(candidate, lam * step_size)

        if np.linalg.norm(bk - bkt) <= 1e-6 * (1.0 + np.linalg.norm(bkt)):
            break


    if fit_intercept:
        return bk[:-1], bk[-1]
    return bk
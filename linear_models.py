import numpy as np

class Regressor:
    def __init__(self, fit_intercept = True) -> None:
        self.weights = None
        self.fit_intercept = fit_intercept
        pass

    def predict(self, X):
        if self.fit_intercept:
            return np.inner(self.weights, np.c_[np.ones((X.shape[0], 1)), X])
        else:
            return np.inner(self.weights, X)


class LogisticRegression(Regressor): 
    def __init__(self, penalty = 'L2', fit_intecept = True):
        pass


class LinearRegression(Regressor): 
    def __init__(self, loss = 'mse', gamma = 0.1,  fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.weights = None
        self.gamma = gamma
        pass

    def gradient(self, X, y, w):
        return -1/X.shape[0]*np.inner(X.T, y - np.inner(X, w))

    def fit(self, X, y, initial_w, max_iters):
        if self.fit_intecept:  
            X = np.c_[np.ones((X.shape[0], 1)), X]
        w = initial_w

        for _ in range(max_iters):
            w -= self.gamma * self.gradient(y,X,w)
            
        return w, MSE(y, np.inner(w, X))


class RidgeRegression(Regressor):
    def __init__(self, alpha = 1, fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.alpha = alpha
        self.weights = None
        pass

    def fit(self, X, y):
        if self.fit_intecept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.inner(np.dot(np.linalg.inv(np.dot(X.T, X) + self.alpha * 2 * X.shape[0] * np.eye(X.shape[1])), X.T), y)
        return self.weights, MSE(y, np.inner(self.weights, X))
        

class RidgeRegression(Regressor):
    def __init__(self, fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.weights = None
        pass

    def fit(self, X, y):
        if self.fit_intecept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.inner(np.dot(np.linalg.inv(np.dot(X.T, X) + self.alpha * 2 * X.shape[0] * np.eye(X.shape[1])), X.T), y)
        return self.weights, MSE(y, np.inner(self.weights, X))


def least_squares(y, tx, offset = False):
    
    # Adding offset
    if offset:  
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    
    
    B = np.inner(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)
    return B, MSE(y, np.inner(B, tx))
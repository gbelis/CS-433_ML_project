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
    
    def mse(self, X, y):
        return 0.5 * np.average((y - np.inner(X, self.weights)) ** 2)
    
    def mae(self, X, y):
        return np.average(np.abs(y - np.inner(self.weights, X)))
    
    def get_weights(self):
        return self.weights


class LogisticRegression(Regressor): 
    def __init__(self, penalty = 'L2', fit_intecept = True):
        pass


class LinearRegression(Regressor): 
    def __init__(self, loss = 'mse', gamma = 0.1,  fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.weights = None
        self.gamma = gamma
        self.loss = loss
        pass

    def mse_gradient(self, X, y):
        return -1/X.shape[0]*np.dot(X.T, y - X.dot(self.weights))
    

    def fit(self, X, y, initial_w, max_iters, batch = None):
        if self.fit_intecept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = initial_w.reshape((X.shape[1],1))
        if not batch:
            batch = X.shape[0]

        for _ in range(max_iters):
            idx = np.random.choice(np.arange(X.shape[0]), replace = False, size=batch)
            self.weights -= self.gamma * self.get_gradient(X[idx],y[idx])
        return self
    
    def get_gradient(self, X,y):
        if self.loss == 'mse':
            return self.mse_gradient(X,y)
        elif self.loss == 'mae':
            return self.mae_gradient(X,y)

    def get_loss(self, X, y):
        if self.loss == 'mse':
            return self.mse(X, y)
        elif self.loss == 'mae':
            return self.mae(X, y)


class RidgeRegression(Regressor):
    def __init__(self, alpha = 1, fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.alpha = alpha
        self.weights = None
        pass

    def fit(self, X, y):
        if self.fit_intecept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.solve(np.dot(np.dot(X.T, X) + self.alpha * 2 * X.shape[0] * np.eye(X.shape[1]), X.T), y)
        return self
        


class LtsqrRegression(Regressor):
    def __init__(self, fit_intecept = True):
        self.fit_intecept = fit_intecept
        self.weights = None
        pass

    def fit(self, X, y):
        if self.fit_intecept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.solve(np.dot(X.T, X), np.inner(X.T, y))
        return self


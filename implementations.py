from linear_models import *

def MAE(y, pred):
    return np.average(np.abs(y - pred))

def MSE(y, pred):
    return 0.5 * np.average((y - pred) ** 2)

def MSE_gradient(y, tx, w):
    return -1/tx.shape[0]*np.inner(tx.T, y - np.inner(tx, w))

def least_squares(y, tx, fit_intecept = False):
    model = LtsqrRegression(fit_intecept = fit_intecept).fit(tx, y)
    return model.weights, model.mse(tx, y)

# Linear regression, MSE Loss, Gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, fit_intecept = False):
    model = LinearRegression(fit_intecept = fit_intecept, gamma=gamma).fit(tx, y, initial_w = initial_w, max_iters=max_iters)
    return model.weights, model.mse(tx, y)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, fit_intecept = False):
    model = LinearRegression(fit_intecept = fit_intecept, gamma=gamma).fit(tx, y, initial_w = initial_w, max_iters=max_iters)
    return model.weights, model.mse(tx, y)

def ridge_regression(y, tx, lambda_ = 1, fit_intecept = False):
    model = RidgeRegression(fit_intecept = fit_intecept, alpha = lambda_).fit(tx, y)
    return model.weights, model.mse(tx, y)





def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, offset = False, hist = False):
        
    # Adding offset
    if offset:  
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    B = initial_w
    hist = []

    for _ in range(max_iters):
        B -= gamma * reg_log_gradient(y, tx, B, lambda_)
        hist.append(MSE(y, np.inner(B, tx)))
    if hist:
        return B, MSE(y, np.inner(B, tx)), hist

    return B, MSE(y, np.inner(B, tx))



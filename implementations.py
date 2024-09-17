import numpy as np

def MAE(y, pred):
    return np.average(np.abs(y - pred))

def MSE(y, pred):
    return 0.5 * np.average((y - pred) ** 2)

def MSE_gradient(y, tx, w):
    return -1/tx.shape[0]*np.inner(tx.T, y - np.inner(tx, w))

def least_squares(y, tx, offset = False):
    
    # Adding offset
    if offset:  
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    
    
    B = np.inner(np.dot(np.linalg.inv(np.dot(tx.T, tx)), tx.T), y)
    return B, MSE(y, np.inner(B, tx))

def ridge_regression(y, tx, lambda_, offset = False):
    if offset:
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    print(tx.shape)
    B = np.inner(np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_/(2*tx.shape[0]) * np.eye(tx.shape[1])), tx.T), y)
    return B, MSE(y, np.inner(B, tx))


# Linear regression, MSE Loss, Gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, offset = False):
    
    # Adding offset
    if offset:  
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    B = initial_w
    hist = []

    
    for _ in range(max_iters):
        B -= gamma * MSE_gradient(y,tx,B)
        hist.append(MSE(y, np.inner(B, tx)))

    return B, MSE(y, np.inner(B, tx)), hist

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch = 2, offset = False):
    
    # Adding offset
    if offset:  
        tx = np.c_[np.ones((tx.shape[0], 1)), tx]
    B = initial_w
    hist = []

    
    for _ in range(max_iters):
        idx = np.random.randint(tx.shape[0], size=batch)
        tx_, y_ = tx[idx], y[idx]
        B -= gamma * MSE_gradient(y_,tx_,B)
        hist.append(MSE(y, np.inner(B, tx)))

    return B, MSE(y, np.inner(B, tx)), hist

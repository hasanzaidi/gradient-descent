# A simple vectorised implementation of batch and stochastic gradient descent
import numpy as np
import time

# Hyperparameters
alpha = 0.01
epochs = 3000


def batch_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))

        # Print thetas every 500 iterations to show progress
        if it % 500 == 0:
            print("Batch thetas after {} iterations: {}, {}".format(it, theta[0], theta[1]))

    return theta


def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for it in range(iterations):
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind, :].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)

            prediction = np.dot(X_i, theta)
            theta = theta - (1 / m) * learning_rate * (X_i.T.dot((prediction - y_i)))

        # Print thetas every 500 iterations to show progress
        if it % 500 == 0:
            print("Stochastic thetas after {} iterations: {}, {}".format(it, theta[0], theta[1]))

    return theta


# Generate input and output data following model y = -4x + 70
m = 1000
X = 2 * np.random.rand(m, 1)
y = 70 - 4 * X + np.random.randn(m, 1)

# Add 1s for theta 0
X_b = np.c_[np.ones((len(X), 1)), X]

start = time.time()
theta = np.random.randn(2, 1)
theta = stochastic_gradient_descent(X_b, y, theta, alpha, epochs)
stop = time.time()
print('\nStochastic results took: {:0.1f} seconds'.format(stop - start))
print('Theta0:          {:0.3f}\nTheta1:          {:0.3f}\n'.format(theta[0][0], theta[1][0]))

start = time.time()
theta = np.random.randn(2, 1)
theta = batch_gradient_descent(X_b, y, theta, alpha, epochs)
stop = time.time()
print('\nBatch results took: {:0.1f} seconds'.format(stop - start))
print('Theta0:          {:0.3f}\nTheta1:          {:0.3f}'.format(theta[0][0], theta[1][0]))

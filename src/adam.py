import math

# Hyperparameters
alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8


# Function we are trying to minimise (x^2 - 4x + 6)
def func(x):
    return x**2 - (4 * x) + 6


# Derivative of the function (2x - 4)
def grad_func(x):
    return 2 * (x - 4)


# Initialise data
theta = 4.1
m_t = 0
v_t = 0
t = 0

# Loop keeps running until we break out of the loop
while 1:
    # Time
    t += 1

    gradient = grad_func(theta)

    # First moment (mean of the gradient)
    m_t = beta_1 * m_t + (1 - beta_1) * gradient

    # Second moment (uncentered variance of the gradient)
    v_t = beta_2 * v_t + (1 - beta_2) * (gradient * gradient)

    # Bias corrected first moment
    m_hat = m_t / (1 - (beta_1 ** t))

    # Bias corrected second moment
    v_hat = v_t / (1 - (beta_2 ** t))

    theta_0_prev = theta

    # Update theta
    theta = theta - (alpha * m_hat) / (math.sqrt(v_hat) + epsilon)

    # Print theta every 10 iterations
    if t % 100 == 0:
        print("Theta after {} iterations: {}".format(t, theta))

    # Stops if we have converged
    if theta == theta_0_prev:
        print("\nConverged after {} iterations".format(t))
        break


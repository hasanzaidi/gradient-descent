# A simple non-vectorised implementation of batch gradient descent
# This uses data below which only has one feature and thus can be represented by y = -4x + 70

# Hyperparameters
alpha = 0.01
thetas = [0, 0]
epochs = 5000

# Generate training data
desired_thetas = [70, -4]
m = 25
x0 = [1 for num in range(m)]
x1 = [num for num in range(m)]
y = [(desired_thetas[1] * num) + desired_thetas[0] for num in x1]

for e in range(epochs):
    # Theta 0
    total_gradient = 0
    for i in range(m):
        h = (thetas[0] * x0[i]) + (thetas[1] * x1[i])
        gradient = (h - y[i]) * x0[i]
        total_gradient = gradient + total_gradient
    thetas[0] = (thetas[0] - (alpha * (total_gradient / m)))

    # Theta 1
    total_gradient = 0
    for i in range(m):
        h = (thetas[0] * x0[i]) + (thetas[1] * x1[i])
        gradient = (h - y[i]) * x1[i]
        total_gradient = gradient + total_gradient
    thetas[1] = (thetas[1] - (alpha * (total_gradient / m)))

    # Print thetas every 1000 iterations
    if e % 1000 == 0:
        print("Thetas after {} iterations: {:.3f}, {:.3f}".format(e, thetas[0], thetas[1]))

print("\nFinal estimate for value which should be 70: {:.3f}".format(thetas[0]))
print("Final estimate for value which should be -4: {:.3f}".format(thetas[1]))

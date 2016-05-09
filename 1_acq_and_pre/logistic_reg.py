import numpy as np
import random


# m denotes the number of examples here, not the number of features
def gradient_descent(x, y, theta, alpha, m, num_iterations):
    x_trians = x.transpose()
    for i in xrange(0, num_iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss**2)/(2*m)
        print "Iteration %d / Cost: %f" % (i, cost)
        # avg gradient per example
        gradient = np.dot(x_trians, loss)/m
        # update
        theta = theta - alpha * gradient
    return theta


def gen_data(num_points, bias, variance):
    x = np.zeros(shape=(num_points, 2))
    y = np.zeros(shape=num_points)
    # basically a straight line
    for i in xrange(0, num_points):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i+bias)+random.uniform(0, 1)*variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = gen_data(100, 25, 10)
print "x:"
print x
print "y:"
print y
m, n = np.shape(x)
num_iterations = 100
alpha = 0.0005
theta = np.ones(n)
theta = gradient_descent(x, y, theta, alpha, m, num_iterations)
print theta

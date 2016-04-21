import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in xrange(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i-1] + 1, layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i+1]))-1)*0.25)
        # weights=[array([[-0.16051946,  0.18211593,  0.21787758],
        #                 [ 0.09864937,  0.15279863,  0.1243071 ],
        #                 [-0.14374152,  0.12805746,  0.1004326 ]]),                 array([[-0.19295282],
        #                 [-0.04271876],
        #                 [ 0.02968057]])]

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in xrange(epochs):
            i = np.random.randint(X.shape[0])
            print self.weights,X,i
            # take the random line in X
            a = [X[i]]

            for l in xrange(len(self.weights)):
                # the same of the columns of a and the length of weights
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            # i=2;a=[array([ 1.,  0.,  1.]), array([-0.29520713,  0.30059483,  0.30797818]), array([ 0.05321067])]
            error = y[i] - a[-1]
            # error5=array([ 0.94678933])
            deltas = [error * self.activation_deriv(a[-1])]
            # deltas=[(1-O5)*(1-O5^2)] [array([ 0.94411366])]

            #Staring backpropagation
            for l in xrange(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            # deltas=[array([ 0.94411366]), array([-0.16717261, -0.03689592,  0.02552341])]
            deltas.reverse()
            for i in xrange(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in xrange(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

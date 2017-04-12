import numpy as np


class LinearRegression:

    def __init__(self):
        pass

    @staticmethod
    def compute_cost(x, y, theta=None):
        if theta is None:
            theta = np.zeros(x.shape[1]).reshape(-1, 1)

        m = float(y.size)
        h = x.dot(theta)
        j = (1 / (2 * m)) * np.sum(np.square(h - y))

        return j

    @classmethod
    def gradient_descent(cls, x, y, theta=None, alpha=0.01, iters_num=700):
        if theta is None:
            theta = np.zeros(x.shape[1]).reshape(-1,1)

        m = float(y.size)
        j_seq = np.zeros(iters_num)
        theta_seq = np.zeros((iters_num, x.shape[1]))

        for itr in np.arange(iters_num):
            h = x.dot(theta)
            theta -= alpha * (1 / m) * (x.T.dot(h - y))
            theta_seq[itr] = theta.T
            j_seq[itr] = cls.compute_cost(x, y, theta)

        return theta, theta_seq, j_seq


class LogisticRegression:

    def __init__(self):
        pass

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @classmethod
    def compute_cost(cls, theta, x, y):
        m = float(y.size)
        h = cls.sigmoid(x.dot(theta))
        j = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

        if np.isnan(j[0]):
            return np.inf
        return j[0]

    @classmethod
    def gradient(cls, theta, x, y):
        m = float(y.size)
        h = cls.sigmoid(x.dot(theta.reshape(-1, 1)))
        grad = (1 / m) * x.T.dot(h - y)

        return grad.flatten()


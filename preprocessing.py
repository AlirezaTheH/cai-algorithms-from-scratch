import numpy as np


def normalize_data(x):
    """Normalizing such that data has mean 0 and std of 1"""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    x = (x - mu) / sigma
    return x


def map_feature(x, degree):
    # Number of features
    n = x.shape[1]

    # Reshape 1, n, number of examples
    x = x.T[np.newaxis]

    # Create powers from 0 to degree for each feature
    powers = np.tile(np.arange(degree + 1), (n, 1)).T[..., np.newaxis]

    # Couples of powers from features
    all_power_couples = np.indices((degree + 1,) * n).reshape(n, (degree + 1) ** n).T

    power_matrix = np.power(x, powers)
    required_couples = all_power_couples[:, 0] + all_power_couples[:, 1] <= degree
    result = np.product(np.diagonal(power_matrix[all_power_couples[required_couples]], 0, 1, 2), axis=2)

    return result.T

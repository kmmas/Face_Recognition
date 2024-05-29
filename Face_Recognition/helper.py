import numpy as np
def centralize_data(data_matrix) :
    row_means = np.mean(data_matrix, axis=0)
    Z = data_matrix - row_means
    return Z

def projected_data_calculation(Z,U) :
    projected_data = np.dot(Z, U)
    return projected_data

def order_eigens(eigen_values,eigen_vectors) :
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return eigen_values, eigen_vectors
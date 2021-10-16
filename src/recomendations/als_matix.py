import numpy as np

def als_matrix( matrix, second_shape_first_matrix, n_iteration, not_null_values, learning_rate = 1, init_bias = 0.3, init_var = 0.15 ):
    """ ALS matrix factorization algorithm

    Args:
        matrix ([np.array]): [interaction matrix for user-item]
        second_shape_first_matrix ([int]): [shape for choice matrix "u" dimation ]
        learning_rate ([float]): [gradient descent step]
        n_iteration ([int]): [count of iteration]
        not_null_values ([type]): [known values]
        init_bias (float, optional): [normal distribution bias for initializing initial matrices]. Defaults to 0.3.
        init_var (float, optional): [normal distribution variance for initializing initial matrices]. Defaults to 0.15.

    Returns:
        [np.array]: [estimated matrix]

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> user_item   = np.array( [[1,0,0,1], [1,1,0,0], [0,0,0,1]] )

        >>> als_matrix( user_item, 2, 400, 1)
        array([[0.98981623, 0.74652651, 0.25153983, 1.00197617],
               [0.99896439, 0.98994769, 0.32799586, 0.86552836],
               [0.90507976, 0.56661684, 0.19364865, 0.98766064]])
        
    """

    # размерность исходной матрицы для инициализации матриц "u" и "v"
    first_shape, second_shape = matrix.shape
    u = np.random.normal( init_bias, init_var, (first_shape, second_shape_first_matrix))
    v = np.random.normal( init_bias, init_var, (second_shape_first_matrix, second_shape))

    # learning rate для известных значений
    not_null_err = 0.05
    # learning rate для неизвесных значений ( берем маленькое число, чтобы не занулить неизвестные значения )
    null_err = 0.0005
    # коэффициент регуляризации 
    reg_coef = 0.001

    # инициализируем ошибку
    err = 0

    for step in range(n_iteration):

        # ошибка оценки
        eps = np.dot(u, v) - matrix

        # фиксируем "v", считаем градиент по "u"
        for i, u_item in enumerate(u):
            err = 0
            for j, v_item in enumerate(v.T):
                # градиент
                err_form = eps[i][j] * v_item + reg_coef * v_item
                # если это извесное значение, умножаем ошибку на learning rate для извесны значений
                if matrix[i][j] == not_null_values:
                    err += not_null_err*err_form
                else:
                    err += null_err * err_form

            u[i] -= learning_rate * err

        # фиксируем "u", считаем градиент по "v"
        v_t = v.T
        for j, u_item in enumerate(v.T):
            err = 0
            for i, v_item in enumerate(u):
                err_form = eps[i][j] * v_item + reg_coef * v_item
                if matrix[i][j] == not_null_values:
                    err += not_null_err*err_form
                else:
                    err += null_err * err_form

            v_t[j] -= learning_rate * err
        v = v_t.T

    return np.dot(u, v)

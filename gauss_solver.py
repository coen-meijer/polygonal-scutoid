import numpy as np

def pivot_succes(matrix, column):
    index_largest = column
    maximum = abs(matrix[column, column])
    for index in range(column + 1, len(matrix)):
        if abs(matrix[index][column]) > maximum:
            maximum = abs(matrix[index][column])
            index_largest = index
    if column != index_largest:
        matrix[[column, index_largest]] = matrix [[index_largest, column]]  #swap
    # return succes of failure
    return maximum > 0


def gauss_solve_lower(matrix):
    for diagonal_index in range(matrix.shape[0]):
        if(pivot_succes(matrix, diagonal_index)):
            diagonal_val = matrix[diagonal_index, diagonal_index]
            matrix[diagonal_index] /= diagonal_val
            for row_index in range(diagonal_index + 1, matrix.shape[0]):
                elimination_val = matrix[row_index, diagonal_index]
                matrix[row_index] -= elimination_val * matrix[diagonal_index]
        else:
            return False
    return True


def gauss_solve_upper(matrix):
    for diagonal_index in range(matrix.shape[0] - 1, -1, -1):
        for column_index in range(matrix.shape[0] - 1, diagonal_index, -1):
            elimination_val = matrix[diagonal_index, column_index]
            matrix[diagonal_index] -= elimination_val * matrix[column_index]


def gauss_solve(matrix):
    if (gauss_solve_lower(matrix)):
        gauss_solve_upper(matrix)
        return True
    else:
        return False

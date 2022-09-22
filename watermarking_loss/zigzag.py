import numpy as np
from .helpers import flatten


def apply_zigzag(data: np.ndarray, rows=8, columns=8):
    """
    This function performs the zigzag traverse.
    Refer https://www.geeksforgeeks.org/print-matrix-in-zig-zag-fashion/

    :param data: 2D array with input data.
    :param rows: array shape size of the x-axis.
    :param columns: array shape size pf the y-axis.
    :return: Returns the traversed zigzag matrix.
    """

    solution = [[] for i in range(rows + columns - 1)]

    for i in range(rows):
        for j in range(columns):
            sum = i + j
            if((sum % 2) == 0):
                solution[sum].insert(0, data[i][j])  # add at beginning.
            else:
                solution[sum].append(data[i][j])  # add at end.

    return flatten(solution)

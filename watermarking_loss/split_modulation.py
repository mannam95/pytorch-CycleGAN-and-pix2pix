import numpy as np
from .svd import apply_svd, modify_singular_values


def get_dct_split_modulation_matrix(data: np.ndarray):
    """
    This function gets the split-modiulations matrix

    :param data: a list of any dimensions, as per paper (8*8)
    :return: Returns a tuple of two split modulation matrices
    """
    split_matrix1 = [
                    [data[2][2], data[0][5]],  # 13, 16 from the paper
                    [data[3][2], data[4][1]],  # 19, 20 from the paper
    ]

    split_matrix2 = [
                    [data[1][4], data[2][3]],  # 17, 18 from the paper
                    [data[5][0], data[3][3]],  # 21, 25 from the paper
    ]

    return (split_matrix1, split_matrix2)


def update_dct_split_modulation_matrix(data: np.ndarray, split_matrix1, split_matrix2):
    """
    This function gets the split-modiulations matrix

    :param data: a list of any dimensions, as per paper (8*8)
    :param split_matrix1: first split modulation matrix
    :param split_matrix2: second split modulation matrix
    :return: Returns a tuple of two split modulation matrices
    """
    # update first split modulation matrix
    data[2][2] = split_matrix1[0][0]  # 13
    data[0][5] = split_matrix1[0][1]  # 16
    data[3][2] = split_matrix1[1][0]  # 18
    data[4][1] = split_matrix1[1][1]  # 20

    # update second split modulation matrix
    data[1][4] = split_matrix2[0][0]  # 17
    data[2][3] = split_matrix2[0][1]  # 18
    data[5][0] = split_matrix2[1][0]  # 21
    data[3][3] = split_matrix2[1][1]  # 25

    return data


def update_split_modulation_matrices(watermark_encrypted_bit, split_matrix1, split_matrix2, alpha):
    """
    This function does modification of the split modulation matrices

    :param watermark_encrypted_bit: encrypted watermark bit
    :param split_matrix1: first split modulation matrix
    :param split_matrix2: second split modulation matrix
    :param alpha: embedding strength.
    :return: Returns the modified split modulation matrices
    """

    u1, s1, v1 = apply_svd(split_matrix1)  # Apply svd on split modulation matrix1
    u2, s2, v2 = apply_svd(split_matrix2)  # Apply svd on split modulation matrix2

    new_s1, new_s2 = modify_singular_values(s1, s2, watermark_encrypted_bit, alpha)

    new_sub_mod1 = u1 * new_s1 * v1.transpose()
    new_sub_mod2 = u2 * new_s2 * v2.transpose()

    return new_sub_mod1, new_sub_mod2

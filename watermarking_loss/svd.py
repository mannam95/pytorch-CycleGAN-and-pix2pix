import numpy as np


def apply_svd(data: np.ndarray):
    """
    This function does svd

    :param data: 2D array with input data.
    :return: Returns the u,s,v matrices.
    """
    u, s, v = np.linalg.svd(data, full_matrices=True)
    return (u, s, v)


def modify_singular_values(s1, s2, watermark_encrypted_bit, alpha):
    """
    This function modifies the largest singular values

    :param s1: first singular values of split modulation matrix.
    :param s2: second singular values of split modulation matrix.
    :param watermark_encrypted_bit: encrypted watermark bit.
    :param alpha: embedding strength.
    :return: Returns the modified singular values
    """
    largest_s1_index = list(s1).index(max(list(s1)))
    largest_s2_index = list(s2).index(max(list(s2)))

    maxs1, maxs2 = s1[largest_s1_index], s1[largest_s2_index]  # In paper termed as lambda1, theta1 largest singular values.
    mean_e = (maxs1 + maxs2) / 2  # In paper termed as E.

    new_maxs1, new_maxs2 = None, None

    if watermark_encrypted_bit == 1:
        new_maxs1 = mean_e * alpha
        new_maxs2 = mean_e / alpha
    else:
        new_maxs1 = mean_e / alpha
        new_maxs2 = mean_e * alpha

    # Modify the singular values
    s1[largest_s1_index] = new_maxs1
    s2[largest_s2_index] = new_maxs2

    return (s1, s2)


def get_largest_singular_values(sub_mod1, sub_mod2):
    """
    This function gets the largest singular values

    :param s1: first split modulation matrix
    :param s2: second split modulation matrix
    :return: Returns the largest singular values
    """

    u1, s1, v1 = apply_svd(sub_mod1)  # Apply svd on split modulation matrix1
    u2, s2, v2 = apply_svd(sub_mod2)  # Apply svd on split modulation matrix2

    return (max(list(s1)), max(list(s2)))

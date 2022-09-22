import scipy.fftpack
import numpy as np

from .split_modulation import get_dct_split_modulation_matrix, update_dct_split_modulation_matrix, update_split_modulation_matrices
from .svd import get_largest_singular_values

# Refer https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
# Refer https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html


def dct_2d(data):
    """
    This function does 2D Discrete Cosine Transform(DCT).

    :param data: 2D array with input data.
    :return: Return the Discrete Cosine Transform.
    """
    return scipy.fftpack.dct(scipy.fftpack.dct(data, axis=0, norm='ortho'), axis=1, norm='ortho')


def apply_2d_dct_all_blocks(data: np.ndarray):
    """
    This function apply 2d-dct for each given block.

    :param data: nD array with input data.
    :return: Returns all the blocks which have gone through 2d-dct
    """
    dct_blocks = np.zeros(data.shape)

    for i, idata in enumerate(data):
        for j, jdata in enumerate(idata):
            dct_blocks[i][j] = dct_2d(jdata)
    return dct_blocks


def idct_2d(data):
    """
    This function does 2D inverse Discrete Cosine Transform(DCT).

    :param data: 2D array with dct coefficient matrix.
    :return: Return the original data by performing inverse Discrete Cosine Transform.
    """
    return scipy.fftpack.idct(scipy.fftpack.idct(data, axis=0, norm='ortho'), axis=1, norm='ortho')


def apply_inverse_2d_dct_all_blocks(data: np.ndarray):
    """
    This function apply 2d-dct for each given block.

    :param data: nD array with input data.
    :return: Returns all the blocks which have gone through 2d-dct
    """
    idct_blocks = np.zeros(data.shape)

    for i, idata in enumerate(data):
        for j, jdata in enumerate(idata):
            idct_blocks[i][j] = idct_2d(jdata)
    return idct_blocks


def update_dct_block(current_dct_block, watermark_encrypted_bit, alpha):
    """
    This function performs the updation of a single dct block

    :param current_dct_block: current single dct block
    :param watermark_encrypted_bit: encrypted watermark bit
    :param alpha: embedding strength.
    :return: Returns the modified dct block
    """
    sub_mod1, sub_mod2 = get_dct_split_modulation_matrix(current_dct_block)  # get split modulation matrices
    new_sub_mod1, new_sub_mod2 = update_split_modulation_matrices(watermark_encrypted_bit, sub_mod1, sub_mod2, alpha)  # update split modulation matrices
    updated_dct_block = update_dct_split_modulation_matrix(current_dct_block, new_sub_mod1, new_sub_mod2)  # update current dct block coefficient matrix.

    return updated_dct_block


def update_dct_blocks(dct_blocks, watermark_encrypted, alpha):
    """
    This function performs the updation of dct blocks

    :param dct_blocks: all the current image dct blocks
    :param watermark_encrypted: encrypted watermark
    :param alpha: embedding strength.
    :return: Returns the modified dct blocks
    """
    updated_dct_blocks = np.zeros(dct_blocks.shape)

    for i, idata in enumerate(dct_blocks):
        for j, jdata in enumerate(idata):
            current_dct_block_updated = update_dct_block(jdata, watermark_encrypted[i][j], alpha)  # get updated dct current block
            updated_dct_blocks[i][j] = current_dct_block_updated  # store the updated dct block

    return updated_dct_blocks


def get_single_dct_block_watermarkbit(dct_block):
    """
    This function performs the extraction of the single watermark bit

    :param dct_block: single dct block
    :return: Returns the extracted watermark bit
    """

    sub_mod1, sub_mod2 = get_dct_split_modulation_matrix(dct_block)  # get split modulation matrices

    ls1, ls2 = get_largest_singular_values(sub_mod1, sub_mod2)

    if ls1 >= ls2:
        return 1
    else:
        return 0


def get_watermarkbits_from_dct_blocks(dct_blocks):
    """
    This function performs the extraction of the watermark bits

    :param dct_blocks: all the current image dct blocks
    :return: Returns the extracted watermark bits
    """
    watermark_bits = np.zeros((dct_blocks.shape[0], dct_blocks.shape[1]))

    for i, idata in enumerate(dct_blocks):
        for j, jdata in enumerate(idata):
            current_dct_block_updated = get_single_dct_block_watermarkbit(jdata)  # get updated dct current block
            watermark_bits[i][j] = current_dct_block_updated  # store the updated dct block

    return watermark_bits

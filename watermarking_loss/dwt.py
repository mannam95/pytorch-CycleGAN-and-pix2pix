import pywt

# Refer https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html


def dwt_2d(data, wavelet="haar"):
    """
    This function does 2D forward Discrete Wavelet Transform(DWT).

    :param data: 2D array with input data.
    :param wavelet: Wavelet object or name string, or 2-tuple of wavelets. This can also be a tuple containing a wavelet to apply along each axis in axes.
    :return: Approximation(LL), horizontal detail(LH), vertical detail(HL) and diagonal detail(HH) coefficients respectively.
    """
    return pywt.dwt2(data, wavelet)


def idwt_2d(coeffs, wavelet="haar"):
    """
    This function does 2D inverse Discrete Wavelet Transform(DWT).
    Reconstructs data from coefficient arrays.

    :param coeffs: Approximation(LL), horizontal detail(LH), vertical detail(HL) and diagonal detail(HH) coefficients respectively.
    :param wavelet: Wavelet object or name string, or 2-tuple of wavelets. This can also be a tuple containing a wavelet to apply along each axis in axes.
    :return: Approximation(LL), horizontal detail(LH), vertical detail(HL) and diagonal detail(HH) coefficients respectively.
    """
    return pywt.idwt2(coeffs, wavelet)

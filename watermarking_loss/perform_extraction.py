import numpy as np
from .dct import apply_2d_dct_all_blocks, get_watermarkbits_from_dct_blocks
from .dwt import dwt_2d
from .helpers import create_non_overlapping_blocks
from util import util
from .watermark_encryption_decryption import Encrypt_Decrypt

class Extract_WaterMark():
    """This class defines config options.

    It also implement a function which will integrate the extracting.
    """

    def __init__(self, options):
        """Init the class."""
        self.options = options
        self.encrypt_decrypt = Encrypt_Decrypt(options)


    def extract_watermark(self, img_tensor):
        """
        This function does all the integration for watermark extraction

        :param img: image which watermark needs to be extracted.
        :return: Returns the extracted watermark image
        """
        extracted_watermarks = []
        for cur_img in img_tensor:
            image_numpy = cur_img.cpu().detach().numpy()
            image_numpy = np.squeeze(image_numpy, axis=0)

            # get the coefficients of DWT transform.
            LL, (LH, HL, HH) = dwt_2d(image_numpy, 'haar')
            
            # Get the selected block.
            selected_block = locals()[self.options.dwt_level]

            # get the non-overlapping blocks.
            non_overlapping_blocks = create_non_overlapping_blocks(selected_block, self.options.dct_block_size)

            # applies dct to all the non-overlapping blocks
            dct_blocks = apply_2d_dct_all_blocks(non_overlapping_blocks)

            # get the watermark bits from the dct blocks.
            watermark_bits = get_watermarkbits_from_dct_blocks(dct_blocks)

            # get the original watermark image.
            original_watermark_image = self.encrypt_decrypt.watermark_image_decryption(watermark_bits)

            # Convert the data type to uint8.
            original_watermark_image =  np.uint8(original_watermark_image)

            original_watermark_image = np.expand_dims(original_watermark_image, axis=0)

            # Convert the int values to binary.
            # original_watermark_image = original_watermark_image > 0
            
            extracted_watermarks.append(original_watermark_image)
        extracted_watermarks = np.array(extracted_watermarks).astype(float)
        return extracted_watermarks

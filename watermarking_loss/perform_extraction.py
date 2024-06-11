import numpy as np
from .dct import apply_2d_dct_all_blocks, get_watermarkbits_from_dct_blocks
from .dwt import dwt_2d
from .helpers import create_non_overlapping_blocks
from util import util
from .watermark_encryption_decryption import Encrypt_Decrypt
from concurrent.futures import ProcessPoolExecutor

class Extract_WaterMark():
    """This class defines config options.

    It also implement a function which will integrate the extracting.
    """

    def __init__(self, options):
        """Init the class."""
        self.options = options
        self.encrypt_decrypt = Encrypt_Decrypt(options)

    def get_original_watermark(self):
        """Get the original watermark."""
        org_watermark_img = self.encrypt_decrypt.get_watermark_img()
        org_watermark_img = np.expand_dims(org_watermark_img, axis=0)
        return org_watermark_img
    
    def extract_watermark_single_image(self, cur_img_np, options):
        """
        This function does all the integration for watermark extraction

        :param cur_img: image which watermark needs to be extracted.
        :param options: options for watermark extraction.
        :return: Returns the extracted watermark image
        """
        image_numpy = cur_img_np
        image_numpy = np.squeeze(image_numpy, axis=0)

        # get the coefficients of DWT transform.
        LL, (LH, HL, HH) = dwt_2d(image_numpy, 'haar')
        
        # Get the selected block.
        selected_block = locals()[options.dwt_level]

        # get the non-overlapping blocks.
        non_overlapping_blocks = create_non_overlapping_blocks(selected_block, options.dct_block_size)

        # applies dct to all the non-overlapping blocks
        dct_blocks = apply_2d_dct_all_blocks(non_overlapping_blocks)

        # get the watermark bits from the dct blocks.
        watermark_bits = get_watermarkbits_from_dct_blocks(dct_blocks)

        # get the original watermark image.
        extracted_watermark_image = watermark_bits

        # Convert the data type to uint8.
        extracted_watermark_image = np.uint8(extracted_watermark_image)
        extracted_watermark_image = np.expand_dims(extracted_watermark_image, axis=0)

        return extracted_watermark_image


    def extract_watermark(self, img_tensor):
        """
        This function does all the integration for watermark extraction

        :param img: image which watermark needs to be extracted.
        :return: Returns the extracted watermark image
        """
        org_watermark_img = self.get_original_watermark()

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.extract_watermark_single_image, cur_img.cpu().detach().numpy(), self.options) for cur_img in img_tensor]
            extracted_watermarks_list = [future.result() for future in futures]

        extracted_watermarks_list = np.array(extracted_watermarks_list).astype(float)
        orginal_watermarks_list = np.array([org_watermark_img for _ in extracted_watermarks_list]).astype(float)

        return (orginal_watermarks_list, extracted_watermarks_list)

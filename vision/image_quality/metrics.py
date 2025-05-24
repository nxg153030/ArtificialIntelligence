import os
import cv2
from math import log10, sqrt
import numpy as np


def ssim(original_img, degraded_img):
    pass

def psnr(original_img, degraded_img):
    """
    Peak signal-to-noise ratio expressed in decibels (dB)
    higher the PSNR value, the better.
    :param original_img:
    :param degraded_img:
    :return:
    """
    mse = np.mean((original_img - degraded_img) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel/sqrt(mse))
    return psnr


if __name__ == '__main__':
    print(__file__)
    # path = os.path.join(__file__, '../../data/familypic.jpeg')
    path = '../../data/familypic.jpeg'
    assert os.path.isfile(path)
    img = cv2.imread(path)
    output = psnr(img, img)
    print(f'PSNR: {output}')
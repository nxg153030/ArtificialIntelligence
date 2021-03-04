"""
You're given the red, green and blue channels of an image, that were taken separately.
Assume the displacement is between -30 and 30 pixels
Task: Take the 3 channel image and produce the correctly aligned color image.
"""
from scipy.io import loadmat
import numpy as np
import cv2


def sum_of_squared_differences(u, v):
    return np.linalg.norm(u-v) ** 2


def normalized_cross_correlation(u, v):
    norm_u = u / np.linalg.norm(u)
    norm_v = v / np.linalg.norm(v)
    return np.dot(norm_u, norm_v)


def get_best_shift(constant_channel, channel_to_be_shifted):
    best_shift = temp_ssd = 0
    best_ssd = np.inf
    for i in range(-30, 31):
        temp_shift = i
        shifted_channel = np.roll(channel_to_be_shifted, temp_shift, 0)
        # compute the SSD for each row, and sum it up
        for row in range(len(constant_channel)):
            temp_ssd += sum_of_squared_differences(constant_channel[row], shifted_channel[row])
        if temp_ssd < best_ssd:
            best_ssd = temp_ssd
            best_shift = temp_shift

    return best_shift, best_ssd


def image_aligner(red, blue, green):
    best_ssd_dict = dict()

    # blue -> red alignment
    best_shift_red, best_ssd = get_best_shift(blue, red)
    best_ssd_dict.update({'blue_red': best_ssd})

    # blue -> green alignment
    best_shift_green, best_ssd = get_best_shift(blue, green)
    best_ssd_dict.update({'blue_green': best_ssd})

    blue_aligned_ssd_sum = best_ssd_dict['blue_red'] + best_ssd_dict['blue_green']

    # green -> blue alignment
    best_shift_blue, best_ssd = get_best_shift(green, blue)
    best_ssd_dict.update({'green_blue': best_ssd})

    # green -> red alignment
    best_shift_red, best_ssd = get_best_shift(green, red)
    best_ssd_dict.update({'green_red': (best_shift_red, best_ssd)})

    green_aligned_ssd_sum = best_ssd_dict['green_blue'] + best_ssd_dict['green_red']

    # red -> blue alignment
    best_shift_blue, best_ssd = get_best_shift(red, blue)
    best_ssd_dict.update({'green_blue': (best_shift_blue, best_ssd)})

    # red -> green alignment
    best_shift_green, best_ssd = get_best_shift(red, green)
    best_ssd_dict.update({'green_red': (best_shift_green, best_ssd)})

    aligned_red = np.roll(red, best_shift_red, 0)
    aligned_green = np.roll(green, best_shift_green, 0)
    combined_img = np.dstack((blue, aligned_green, aligned_red))

    # cv2.imshow('image', combined_img)
    # cv2.waitKey()
    cv2.imwrite('../data/img_aligned_blue.png', combined_img)


if __name__ == '__main__':
    red_channel_path = '../data/red.mat'
    green_channel_path = '../data/green.mat'
    blue_channel_path = '../data/blue.mat'
    red_channel = loadmat(red_channel_path)['red']
    green_channel = loadmat(green_channel_path)['green']
    blue_channel = loadmat(blue_channel_path)['blue']
    # print(red_channel)
    # print(green_channel.shape)
    # print(blue_channel.shape)
    # print(red_channel)
    image_aligner(red_channel, blue_channel, green_channel)
    # ssd = sum_of_squared_differences(red_channel, green_channel)
    # print(ssd)



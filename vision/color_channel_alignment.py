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


def get_best_shift(constant_channel, channel_to_be_shifted, axis=0):
    best_shift = temp_ssd = 0
    best_ssd = np.inf
    for i in range(-30, 100):
        temp_shift = i
        shifted_channel = np.roll(channel_to_be_shifted, temp_shift, axis)
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
    best_shift_red, best_ssd = get_best_shift(blue, red, axis=0)
    best_ssd_dict.update({'blue_red': (best_shift_red, best_ssd)})

    # blue -> green alignment
    best_shift_green, best_ssd = get_best_shift(blue, green, axis=0)
    best_ssd_dict.update({'blue_green': (best_shift_green, best_ssd)})

    blue_aligned_ssd_sum = best_ssd_dict['blue_red'][1] + best_ssd_dict['blue_green'][1]

    # green -> blue alignment
    best_shift_blue, best_ssd = get_best_shift(green, blue, axis=0)
    best_ssd_dict.update({'green_blue': (best_shift_blue, best_ssd)})

    # green -> red alignment
    best_shift_red, best_ssd = get_best_shift(green, red, axis=0)
    best_ssd_dict.update({'green_red': (best_shift_red, best_ssd)})

    green_aligned_ssd_sum = best_ssd_dict['green_blue'][1] + best_ssd_dict['green_red'][1]

    # red -> blue alignment
    best_shift_blue, best_ssd = get_best_shift(red, blue, axis=0)
    best_ssd_dict.update({'red_blue': (best_shift_blue, best_ssd)})

    # red -> green alignment
    best_shift_green, best_ssd = get_best_shift(red, green, axis=0)
    best_ssd_dict.update({'red_green': (best_shift_green, best_ssd)})

    red_aligned_ssd_sum = best_ssd_dict['red_blue'][1] + best_ssd_dict['red_green'][1]

    best_alignment_dict = {'blue': blue_aligned_ssd_sum, 'green': green_aligned_ssd_sum, 'red': red_aligned_ssd_sum}
    best_const_channel = max(best_alignment_dict, key=best_alignment_dict.get)
    print(f'Best channel to keep constant: {best_const_channel}')
    if best_const_channel == 'blue':
        aligned_red = np.roll(red, best_ssd_dict['blue_red'][0], 0)
        aligned_green = np.roll(green, best_ssd_dict['blue_green'][0], 0)
        combined_img = np.dstack((blue, aligned_green, aligned_red))
    elif best_const_channel == 'green':
        aligned_red = np.roll(red, best_ssd_dict['green_red'][0], 0)
        aligned_blue = np.roll(blue, best_ssd_dict['green_blue'][0], 0)
        combined_img = np.dstack((aligned_blue, green, aligned_red))
    else:
        aligned_blue = np.roll(blue, best_ssd_dict['red_blue'][0], 0)
        aligned_green = np.roll(green, best_ssd_dict['red_green'][0], 0)
        combined_img = np.dstack((aligned_blue, aligned_green, red))

    # aligned_red = np.roll(red, best_shift_red, 0)
    # aligned_green = np.roll(green, best_shift_green, 0)
    # combined_img = np.dstack((blue, aligned_green, aligned_red))

    # cv2.imshow('image', combined_img)
    # cv2.waitKey()
    cv2.imwrite(f'../data/img_aligned_{best_const_channel}_axis_1.png', combined_img)


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



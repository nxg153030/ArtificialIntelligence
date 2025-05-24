def demosaic(img):
    """
    The process used to reconstruct the image from the incomplete color samples
    of an image sensor overlaid with a color filter array (CFA)
    References:
    https://rawpedia.rawtherapee.com/Demosaicing
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.9866&rep=rep1&type=pdf
    :param img:
    :return:
    Step 1: Reconstruction
    Step 2: Enhancement
    """
    pass


if __name__ == '__main__':
    CFA = [[blue, green], [green, red]]

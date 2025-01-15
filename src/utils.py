import logging
import os

import numpy as np


def get_base_name_wo_postfix(path: str) -> str:
    """Returns the file name in given path without the postfix such as .hdr."""

    base_name = os.path.basename(path).rsplit('.', maxsplit=1)[0]
    return base_name


def img_array_to_rgb(img_array: np.array, possible_R: str, possible_G: str, possible_B: str):

    max_band = img_array.shape[2] - 1
    possibles = [possible_R, possible_G, possible_B]
    # initialize rgb image as a list of separate one channel images
    img_rgb = []

    for i in range(3):

        should_fill, value = infer_runtime_RGB_value(possibles[i])

        if should_fill:
            img_X = np.ones_like(img_array[:, :, 0]) * value
        else:
            img_X = img_array[:, :, np.clip(value, 0, max_band)]

        img_rgb.append(img_X)

    # Stack list of one channel images to make proper numpy array
    img_rgb = np.stack(img_rgb, axis=2)

    """
    Scale the image somehow in hope it will look sensible on the screen

    Take a histogram of the RGB image and use one of the bin edges near the end 
    to scale the pixel values down. This should let some of the brightest pixels (specular 
    reflections) to clip so that they do not make the whole image too dark.              
    """

    logging.info(f"RGB image max value: {np.max(img_rgb)}, median: { np.median(img_rgb)}, mean: {np.mean(img_rgb)}")

    # Arbitrary bin count that seems to work OK
    histogram, bin_edges = np.histogram(img_rgb, bins=10)

    # Arbitrary selection of the cut point. Could be done better using the derivative of the histogram?
    scale = bin_edges[-3]

    # Debugging print out of the bin edges in case the binning needs to be adjusted later.
    # for i,bin_edge in enumerate(bin_edges):
    #     if i > 0:
    #         print(f"Bin edges: {bin_edge} val {histogram[i-1]}")

    logging.info(f"I'll try to scale the RGB image down by {scale:.2f} before gamma correction.")

    img_rgb = img_rgb / scale
    img_rgb = np.sqrt(img_rgb)

    logging.info(f"RGB image after scaling; max value: {np.max(img_rgb)}, median: {np.median(img_rgb)}, mean: { np.mean(img_rgb)}")

    return img_rgb


def infer_runtime_RGB_value(value: str):
    """Proces the string that user has given as band selection or fill value.

    Value is interpreted as band selection if one can directly cast it into an
    integer. Otherwise, if it starts with character 'f' and the rest can be
    cast to integer, the return value is (True, int).

    :returns:
        (bool, int) where the bool indicates if the int should be inferred as a fill value.
        If False, the int should be inferred as a band number.
    """

    should_fill = False
    if value.startswith('f'):
        to_int = value[1:]
        should_fill = True
    else:
        to_int = value

    try:
        parsed_int = int(to_int)
    except ValueError as e:
        print(f"Could not parse fill value from string '{value}'. Input a raw int or for filling an RGB channel with "
              f"a single value, provide string in format fXXX.., where XXX.. can be interpreted as an integer.")
        raise

    return should_fill, parsed_int

import numpy as np


def dark_correction(cube: np.array, dark: np.array, axis):
    """Calculates dark correction for given cube.

    Resulting cube is non-negative.

    :param cube:
        3D numpy array representing the cube.
    :param dark:
        3D numpy array representing the dark cube.
    :param axis:
        Axis along which the dark correction is calculated as numpy axis (int or tuple of ints).
    :return:
        3D numpy array representing the cube with dark correction applied.
    """

    if cube is None:
        raise ValueError(f"Cannot calculate dark because image array is None.")

    if dark is None:
        raise ValueError(f"Cannot calculate dark because dark cube is None.")

    dark_subtractable = np.median(dark, axis=axis)
    cube = cube - dark_subtractable
    cube = np.clip(cube, a_min=0, a_max=None)

    return cube


def white_correction(cube, white_cube, dark_cube, pixelwise=False):
    """Calculates white correction for given cube.

    The dark is subtracted from the white cube before the division.

    :param cube:
        3D numpy array representing the cube.
    :param white_cube:
        3D numpy array representing the white cube.
    :param dark_cube:
        3D numpy array representing the dark cube.
    :param pixelwise:
        If True, the white correction is done pixelwise. Otherwise, the white correction is done using the
        average of the white cube.
    :return:
        3D numpy array representing the cube with white correction applied.
    """

    if cube is None:
        raise ValueError(f"Cannot calculate white because image array is None. Select a cube first.")

    if white_cube is None:
        raise ValueError(f"Cannot calculate white because white cube is None. Select a white cube first.")

    if dark_cube is None:
        raise ValueError(f"Cannot calculate white because dark cube is None. Select a dark cube first.")

    if not pixelwise:
        white_spectrum = np.average(white_cube, axis=(0, 1))
    else:
        white_spectrum = white_cube

    white_spectrum = white_spectrum - np.median(dark_cube, axis=(0, 1))

    cube = np.divide(cube, white_spectrum, dtype=np.float32)

    return cube

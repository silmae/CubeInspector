import unittest
import numpy as np
from numpy.core.defchararray import title
from numpy.ma.testutils import assert_almost_equal
from scipy.ndimage import label
from scipy.stats import mode
import matplotlib.pyplot as plt

from src.test import spectral_mockup as sm
import src.cube_logic as cl




class TestCubeLogic(unittest.TestCase):
    """
    - All spectra should be multiplied to some level as integers to mimic a digital number of the sensor.
    - Create two kinds of spectral signal, say, sine and cosine waves. These are used to create spatial variation.
    - Create dark sensor frames. A single dark frame is the sensor image. For line scanner, it would be one
        spatial and one spectral dimension. The third dimension is essentially the multiple scan lines.
        The values of each sensor pixel should have a known mode as it is used instead of the mean to avoid
        producing floats to the dark corrected cube.
    - Create lamp signal spectrum. This can be half of a sine wave over the whole spectrum. Add dark spectrum to it.
    - Add signal, dark and lamp spectrum together and create a cube.
    - First, subtract dark from the cube and check that the result is lamp + signal
    - Second, subtract dark from white and check that the result is the original lamp white.
    - Third, divide the cube by white and check that the result is the original signal.
    """



    def test_dark_correction(self):
        """
        Create a dark cube with the same size as the spectral cube. This would normally not
        be necessary to have the same shape, but we need it to get the mode to be the same.
        So, a dark cube and its known mode is subtracted from the signal cube. The result should
        be the same as the dark correction method gives when given the cube and the dark cube.
        :return:
        """

        amplitude = 10000
        len_x = 3
        len_y = 4
        len_l = 5
        rand_low = 100
        rand_high = 300
        start = 400
        stop = 2500

        axis = 0
        while axis <= 2:
            cube = sm.create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
            dark_cube, dark_mode, _ = sm.create_dark_cube(len_x=len_x, len_y=len_y, len_l=len_l, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop, axis=axis)
            cube_with_dark_current = cube + dark_cube

            # Perform dark correction
            result = cl.dark_correction(cube_with_dark_current, dark_cube, axis=axis)

            # We'll have to do this weird gymnastics to get the last two dimensions to mach for
            # numpy being able to broadcast operations.
            if axis == 1:
                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 0, 1)
                expected = cube_with_dark_current - dark_mode
                expected = np.swapaxes(expected, 0, 1)
            elif axis == 2:
                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 0, 2)
                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 1, 2)
                expected = cube_with_dark_current - dark_mode
                expected = np.swapaxes(expected, 1, 2)
                expected = np.swapaxes(expected, 0, 2)
            else:
                expected = cube_with_dark_current - dark_mode

            # Assert the result is as expected
            np.testing.assert_array_equal(result, expected)
            axis += 1


if __name__ == '__main__':
    unittest.main()

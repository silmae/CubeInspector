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
        So, a dark cube's known mode is subtracted from the signal cube. The result should
        be the same as the dark correction method gives when given the cube and the dark cube.
        """

        amplitude = 10000
        len_x = 13
        len_y = 14
        len_l = 15
        rand_low = 100
        rand_high = 300
        start = 400
        stop = 2500

        axis = 0
        while axis <= 2:
            cube = sm.create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
            dark_cube, dark_mode, _ = sm.create_dark_cube(len_x=len_x, len_y=len_y, len_l=len_l, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop, axis=axis)
            cube_with_dark_current = cube + dark_cube

            # Check shapes
            self.assertTrue(cube.shape, (len_x, len_y, len_l))
            self.assertTrue(dark_cube.shape, (len_x, len_y, len_l))
            self.assertTrue(cube_with_dark_current.shape, (len_x, len_y, len_l))

            # Check data types
            self.assertTrue(cube.dtype == np.int32)
            self.assertTrue(dark_cube.dtype == np.int32)
            self.assertTrue(cube_with_dark_current.dtype == np.int32)


            # Perform dark correction
            result = cl.dark_correction(cube_with_dark_current, dark_cube, axis=axis)

            """
            We'll have to do this weird gymnastics to get the last two dimensions to mach for
            numpy being able to broadcast operations. There is probably smarter way to do this, 
            but this works, so I'll let it be. When the cube is ordered as (x,y,l) and 
            its shape is (13,14,15) the cases are:
             
            axis == 0
                Cube shape: (13, 14, 15), dark shape: (14, 15). This is the easiest as they can 
                directly broadcast together.
            axis == 1
                Cube shape: (13, 14, 15), dark shape: (13, 15). Swap middle dim to first and then 
                back after subtraction.
            axis == 2
            Cube shape: (13, 14, 15), dark shape: (13, 14). Lots of swapping in similar manner. 
            """
            if axis == 1:

                # print(f"Ax == 1. Cube shape: {cube_with_dark_current.shape}, dark shape: {dark_mode.shape}")
                self.assertTrue(dark_mode.shape, (len_x, len_l))

                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 0, 1)
                expected = cube_with_dark_current - dark_mode
                expected = np.swapaxes(expected, 0, 1)
            elif axis == 2:

                # print(f"Ax == 2. Cube shape: {cube_with_dark_current.shape}, dark shape: {dark_mode.shape}")
                self.assertTrue(dark_mode.shape, (len_x, len_y))

                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 0, 2)
                cube_with_dark_current = np.swapaxes(cube_with_dark_current, 1, 2)
                expected = cube_with_dark_current - dark_mode
                expected = np.swapaxes(expected, 1, 2)
                expected = np.swapaxes(expected, 0, 2)
            else:

                # print(f"Ax == 0. Cube shape: {cube_with_dark_current.shape}, dark shape: {dark_mode.shape}")
                self.assertTrue(dark_mode.shape, (len_y, len_l))

                expected = cube_with_dark_current - dark_mode

            # Assert the result is as expected
            np.testing.assert_array_equal(result, expected)
            axis += 1

        # So far, the data has been complete. Test with handling of negatives, and zeros. Cannot test for NaNs as we have integer
        # arrays and NaNs are a float thing.
        axis = 0
        cube = sm.create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        dark_cube, _, _ = sm.create_dark_cube(len_x=len_x, len_y=len_y, len_l=len_l, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop,
                                                  axis=axis)
        dark_cube[1,1,1] = -1
        dark_cube[2,2,2] = 0

        cube[4, 4, 4] = -1
        cube[5, 5, 5] = 0

        self.assertLessEqual(np.min(dark_cube), 0 ) # has negatives
        self.assertLessEqual(np.min(cube), 0 ) # has negatives

        result = cl.dark_correction(cube, dark_cube, axis=axis)
        self.assertGreaterEqual(np.min(result), 0, msg="Dark correction produced negative values when it should not have.") # no negatives in the result


if __name__ == '__main__':
    unittest.main()

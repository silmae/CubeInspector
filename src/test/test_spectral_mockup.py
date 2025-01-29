import unittest
import numpy as np
from numpy.ma.testutils import assert_almost_equal
from scipy.stats import mode
import matplotlib.pyplot as plt

from src.test import spectral_mockup as sm

DEBUG = False
"""Change to True to show plots for debugging."""

class TestMockup(unittest.TestCase):

    def test_wave(self):
        """Tests that the wave creation function works as expected. """

        n = 20
        m = 2
        start = 200
        stop = 500
        x, wave = sm.create_wave(n=n, m=m, wave="sine", start=start, stop=stop)
        x2, wave2 = sm.create_wave(n=n, m=m, wave="cosine", start=start, stop=stop)

        self.assertEqual(x.shape[0], n)  # Check that the number of points is correct
        self.assertEqual(x[0], start)  # Check that the start value is correct
        self.assertEqual(x[-1], stop)  # Check that the stop value is correct
        self.assertTrue(x.dtype == np.uint64)  # Check that the data type of the x indices is correct
        self.assertLessEqual(np.max(wave), 1)  # Check that the maximum value is less than or equal to 1
        self.assertGreaterEqual(np.min(wave), 0)  # Check that the minimum value is greater than or equal to 0
        self.assertTrue(wave.dtype == np.float64)  # Check that the data type of the wave is correct

        # Same stuff for the cosine wave
        self.assertEqual(x2.shape[0], n)
        self.assertEqual(x2[0], start)
        self.assertEqual(x2[-1], stop)
        self.assertTrue(x2.dtype == np.uint64)
        self.assertLessEqual(np.max(wave2), 1)
        self.assertGreaterEqual(np.min(wave2), 0)
        self.assertTrue(wave2.dtype == np.float64)

        if DEBUG:
            plt.plot(x, wave)
            plt.plot(x2, wave2)
            plt.title("Mock signals")
            plt.show()

    def test_lamp_spectrum(self):
        n = 20
        amplitude = 10000
        start = 100
        stop = 3000
        x, lamp = sm.create_lamp_spectrum(amplitude=amplitude, n=n, start=start, stop=stop)

        self.assertEqual(x.shape[0], n)
        self.assertEqual(x[0], start)
        self.assertEqual(x[-1], stop)
        self.assertTrue(x.dtype == np.uint64)
        self.assertLessEqual(np.max(lamp), amplitude)
        self.assertGreaterEqual(np.min(lamp), 0)
        self.assertTrue(lamp.dtype == np.int64)  # should be int to mimic sensor digital number

        if DEBUG:
            plt.plot(x, lamp)
            plt.title("White spectrum")
            plt.show()

    def test_dark_spectrum(self):
        n = 20
        start = 141
        stop = 698
        rand_low = 987
        rand_high = 13456
        x, dark = sm.create_dark_spectrum(n=n, start=start, stop=stop, rand_low=rand_low, rand_high=rand_high)

        self.assertEqual(x.shape[0], n)
        self.assertEqual(x[0], start)
        self.assertEqual(x[-1], stop)
        self.assertTrue(x.dtype == np.uint64)
        self.assertLessEqual(np.max(dark), rand_high)
        self.assertGreaterEqual(np.min(dark), rand_low)
        self.assertTrue(dark.dtype == np.int32)  # should be int to mimic sensor digital number

        if DEBUG:
            plt.plot(x, dark)
            plt.title("Dark spectrum")
            plt.show()

    def test_dark_cube(self):
        len_x = 100
        len_y = 70
        len_l = 20
        rand_low = 100
        rand_high = 300
        start = 400
        stop = 2500
        axis = 2

        dark_cube, dark_mode, dark_mean = sm.create_dark_cube(len_x=len_x, len_y=len_y, len_l=len_l, rand_low=rand_low, rand_high=rand_high, start=start,
                                                           stop=stop, axis=axis)
        # Check that dimensions and datatype are correct
        self.assertEqual(dark_cube.shape, (len_x, len_y, len_l))
        self.assertEqual(dark_mode.shape, (len_x, len_y))
        self.assertEqual(dark_mean.shape, (len_x, len_y))
        self.assertTrue(dark_cube.dtype == np.int32)
        self.assertTrue(dark_mode.dtype == np.int32)
        self.assertTrue(dark_mean.dtype == np.float32)

        # Test single frame
        dark_frame, dark_mode, dark_mean = sm.create_dark_cube(len_x=len_x, len_y=len_y, len_l=1, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop,
                                                            axis=axis)
        self.assertEqual(dark_frame.shape, (len_x, len_y, 1))
        self.assertIsNone(dark_mode)
        self.assertIsNone(dark_mean)

        if DEBUG:
            plt.imshow(dark_cube[:, :, 0])
            plt.title("Dark cube first band")
            plt.colorbar()
            plt.show()

    def test_white_cube(self):
        len_x = 100
        len_y = 70
        len_l = 50
        amplitude = 10000
        start = 100
        stop = 798

        white_cube = sm.create_white_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(white_cube.shape, (len_x, len_y, len_l))
        self.assertTrue(white_cube.dtype == np.int32)

        # Test single frame
        white_frame = sm.create_white_cube(len_x=len_x, len_y=len_y, len_l=1, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(white_frame.shape, (len_x, len_y, 1))

        if DEBUG:
            plt.imshow(white_cube[:, :, int(len_l / 2)])
            plt.title("White cube middle band")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(white_cube[int(len_x / 2), int(len_y / 2), :])
            plt.title("White cube middle pixel spectrum")
            plt.show()

    def test_distances(self):
        # Test that distances at corners are 1 and in the center 0
        frame = np.zeros((3, 3))
        distances = sm.distances_to_center(frame)
        assert_almost_equal(distances[0][0], 1.0, decimal=16)
        assert_almost_equal(distances[0][2], 1.0, decimal=16)
        assert_almost_equal(distances[2][0], 1.0, decimal=16)
        assert_almost_equal(distances[2][2], 1.0, decimal=16)
        assert_almost_equal(distances[1][1], 0.0, decimal=16)

        # Test that only 2D arrays are accepted
        non_frame = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            sm.distances_to_center(non_frame)
        non_frame = np.zeros((2,))
        with self.assertRaises(ValueError):
            sm.distances_to_center(non_frame)

    def test_signal_cube(self):
        len_x = 250
        len_y = 200
        len_l = 50
        amplitude = 50000
        start = 400
        stop = 2000

        signal_cube = sm.create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(signal_cube.shape, (len_x, len_y, len_l))
        self.assertTrue(signal_cube.dtype == np.int32)

        if DEBUG:
            plt.imshow(signal_cube[:, :, int(len_l * 0.5)])
            plt.title("Signal cube middle band")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(signal_cube[int(len_x / 2), int(len_y / 2), :])
            plt.title("Signal cube middle pixel spectrum")
            plt.show()

    def test_vignetting(self):
        len_x = 250
        len_y = 200
        len_l = 50
        amplitude = 50000
        start = 400
        stop = 2000

        signal_cube = sm.create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        vignetted_cube = sm.kinda_wignetting(signal_cube)

        if DEBUG:
            plt.imshow(vignetted_cube[:, :, int(len_l * 0.5)])
            plt.title("Vignetted signal cube middle band")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(vignetted_cube[int(len_x / 2), int(len_y / 2), :])
            plt.title("Vignetted signal cube middle pixel spectrum")
            plt.show()


if __name__ == '__main__':
    unittest.main()

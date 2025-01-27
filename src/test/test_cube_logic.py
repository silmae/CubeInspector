import unittest
import numpy as np
from numpy.core.defchararray import title
from numpy.ma.testutils import assert_almost_equal
from scipy.ndimage import label
from scipy.stats import mode
import matplotlib.pyplot as plt

import src.cube_logic as cl

DEBUG = True
"""Change to True to show plots for debugging."""

def create_wave(n, m, wave="sine", start=400, stop=2500):
    """
    Create a sine or cosine wave from start to stop with n points, m periods and amplitude from 0 to 1.
    """

    x = np.linspace(0, 2 * np.pi * m, n)
    x_loc = np.linspace(start, stop, n)
    x_loc = x_loc.astype(np.uint64)
    if wave == "sine":
        return x_loc, 0.5 * np.sin(x) + 0.5
    else:
        return x_loc, 0.5 * np.cos(x) + 0.5


def create_lamp_spectrum(amplitude, n, start=400, stop=2500):
    """Generates a quarter of a sine wave with n points and amplitude from 0 to amplitude."""

    x,y = create_wave(n=n, m=0.5, start=start, stop=stop)
    y = 2 * amplitude * y - amplitude
    y = y.astype(np.int64)
    x = x.astype(np.uint64)
    return x, y


def create_dark_spectrum(n, rand_low=100, rand_high=300, start=400, stop=2500):
    """Generates a random signal with n points and values from rand_low to rand_high.

    :param n:
        Number of points in the signal.
    :param rand_low:
        Lower bound for the random values.
    :param rand_high:
        Upper bound for the random values.
    :param start:
        Start wavelength for the signal.
    :param stop:
        Stop wavelength for the signal
    """

    x = np.linspace(start, stop, n)
    y = np.random.randint(rand_low, rand_high, n)
    x = x.astype(np.uint64)
    return x, y

def create_dark_cube(len_x: int, len_y: int, len_l: int, rand_low=100, rand_high=300, start=400, stop=2500, axis=1):
    """Generates a dark frame with len_l spectral points and len_x * len_y spatial points.

    Axes order of the resulting cube is (x, y, l), where l is the spectral axis.

    :param len_x:
        Number of spatial points in the x direction. Setting this to 1 will create a single dark frame in y,l.
    :param len_y:
        Number of spatial points in the y direction. Setting this to 1 will create a single dark frame in x,l.
    :param len_l:
        Number of spectral points in the dark frame. Setting this to 1 will create a single dark frame in x,y.
    :param rand_low:
        Lower bound for the random values.
    :param rand_high:
        Upper bound for the random values.
    :param start:
        Start wavelength for the signal.
    :param stop:
        Stop wavelength for the signal
    :return:
        A dark cube, the mode and the mean of the dark cube. The mode and mean are calculated along the given axis only if the
        length of the dimension is at least 3. If not, the mode and mean ore None.
    """

    dark_cube = np.zeros((len_x, len_y, len_l), dtype=np.int32)
    for i in range(len_x):
        for j in range(len_y):
            x, y = create_dark_spectrum(n=len_l, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop)
            dark_cube[i, j, :] = y

    # Mode and mean only make sense if the given axis is at least 3 long
    if dark_cube.shape[axis] >= 3:
        s_mode,_ = mode(dark_cube, axis=axis) # we don't need the bin count
        dark_mode = np.array(s_mode, dtype=np.int32)
        dark_mean = np.mean(dark_cube, axis=axis, dtype=np.float32)
    else:
        dark_mode = None
        dark_mean = None

    return dark_cube, dark_mode, dark_mean


def create_white_cube(len_x: int, len_y: int, len_l: int, amplitude: int, start=400, stop=2500):
    """Generates a white frame with len_l spectral points and len_x * len_y spatial points.

    Axes order of the resulting cube is (x, y, l), where l is the spectral axis.

    :param len_x:
        Number of spatial points in the x direction. Setting this to 1 will create a single white frame in y,l.
    :param len_y:
        Number of spatial points in the y direction. Setting this to 1 will create a single white frame in x,l.
    :param len_l:
        Number of spectral points in the white frame. Setting this to 1 will create a single white frame in x,y.
    :param amplitude:
        Amplitude of the white signal. The signal is a quarter of a sine wave with amplitude from 0 to amplitude.
    :param start:
        Start wavelength for the signal.
    :param stop:
        Stop wavelength for the signal
    :return:
        A white cube or frame.
    """

    white_cube = np.ones((len_x, len_y, len_l), dtype=np.int32)
    _,lamp_spectrum = create_lamp_spectrum(amplitude=amplitude, n=len_l, start=start, stop=stop)
    white_cube = white_cube * lamp_spectrum
    white_cube = white_cube.astype(np.int32)

    return white_cube


def distances_to_center(frame: np.array) -> np.array:
    """Calculates distance of each pixel to the center of the frame normalized so that the max dist is 1.

    Code adapted from stack overflow discussion:
    From https://stackoverflow.com/questions/66412818/calculate-pixel-distance-from-centre-of-image

    :param frame:
        2D numpy array representing the frame.
    :return:
        2D numpy array representing the distances of each pixel to the center of the frame.
    """

    if len(frame.shape) != 2:
        raise ValueError("The input frame must be 2 dimensional.")

    center = np.array([(frame.shape[0]) / 2, (frame.shape[1]) / 2])
    distances = np.linalg.norm(np.indices(frame.shape) - center[:, None, None] + 0.5, axis = 0)
    dist_max = np.max(distances)
    distances = distances / dist_max

    return distances


def kinda_wignetting(cube: np.array):

    if len(cube.shape) == 3:
        distances = distances_to_center(cube[:, :, 0])
    elif len(cube.shape) == 2:
        distances = distances_to_center(cube)
    else:
        raise ValueError("The input cube must be 2 or 3 dimensional.")

    vignetting = (1 - distances) * 0.5
    if len(cube.shape) == 3:
        cube = np.swapaxes(cube, 0, 2)
        cube = np.swapaxes(cube, 1, 2)
        vignetted_cube = cube * vignetting
        vignetted_cube = np.swapaxes(vignetted_cube, 1, 2)
        vignetted_cube = np.swapaxes(vignetted_cube, 0, 2)
    else:
        vignetted_cube = cube * vignetting

    vignetted_cube = vignetted_cube.astype(np.int32)
    return vignetted_cube


def create_signal_cube(len_x: int, len_y: int, len_l: int, amplitude: int, start=400, stop=2500):
    """Generates a signal cube with len_l spectral points and len_x * len_y spatial points.

    Axes order of the resulting cube is (x, y, l), where l is the spectral axis.

    :param len_x:
        Number of spatial points in the x direction. Setting this to 1 will create a single signal frame in y,l.
    :param len_y:
        Number of spatial points in the y direction. Setting this to 1 will create a single signal frame in x,l.
    :param len_l:
        Number of spectral points in the signal frame. Setting this to 1 will create a single signal frame in x,y.
    :param amplitude:
        Amplitude of the signal. The signal is a quarter of a sine wave with amplitude from 0 to amplitude.
    :param start:
        Start wavelength for the signal.
    :param stop:
        Stop wavelength for the signal
    :return:
        A signal cube.
    """

    signal_cube = np.zeros((len_x, len_y, len_l), dtype=np.int32)
    half_amp = int(amplitude / 2)
    sine_wave = create_wave(n=len_l, m=4, start=start, stop=stop, wave="sine")[1] * half_amp + half_amp
    cosine_wave = create_wave(n=len_l, m=4, start=start, stop=stop, wave="cosine")[1] * half_amp + half_amp
    for i in range(len_x):
        for j in range(len_y):
            if j > i:
                signal_cube[i, j, :] = sine_wave
            else:
                signal_cube[i, j, :] = cosine_wave

    signal_cube = signal_cube.astype(np.int32)
    return signal_cube


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

    def test_wave(self):
        """Tests that the wave creation function works as expected. """

        n = 20
        m = 2
        start = 200
        stop = 500
        x, wave = create_wave(n=n, m=m, wave="sine", start=start, stop=stop)
        x2, wave2 = create_wave(n=n, m=m, wave="cosine", start=start, stop=stop)

        self.assertEqual(x.shape[0], n) # Check that the number of points is correct
        self.assertEqual(x[0], start) # Check that the start value is correct
        self.assertEqual(x[-1], stop) # Check that the stop value is correct
        self.assertTrue(x.dtype == np.uint64) # Check that the data type of the x indices is correct
        self.assertLessEqual(np.max(wave), 1) # Check that the maximum value is less than or equal to 1
        self.assertGreaterEqual(np.min(wave), 0) # Check that the minimum value is greater than or equal to 0
        self.assertTrue(wave.dtype == np.float64) # Check that the data type of the wave is correct

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
        x, lamp = create_lamp_spectrum(amplitude=amplitude, n=n, start=start, stop=stop)

        self.assertEqual(x.shape[0], n)
        self.assertEqual(x[0], start)
        self.assertEqual(x[-1], stop)
        self.assertTrue(x.dtype == np.uint64)
        self.assertLessEqual(np.max(lamp), amplitude)
        self.assertGreaterEqual(np.min(lamp), 0)
        self.assertTrue(lamp.dtype == np.int64) # should be int to mimic sensor digital number

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
        x, dark = create_dark_spectrum(n=n, start=start, stop=stop, rand_low=rand_low, rand_high=rand_high)

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

        dark_cube, dark_mode, dark_mean = create_dark_cube(len_x=len_x, len_y=len_y, len_l=len_l, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop, axis=axis)
        # Check that dimensions and datatype are correct
        self.assertEqual(dark_cube.shape, (len_x, len_y, len_l))
        self.assertEqual(dark_mode.shape, (len_x, len_y))
        self.assertEqual(dark_mean.shape, (len_x, len_y))
        self.assertTrue(dark_cube.dtype == np.int32)
        self.assertTrue(dark_mode.dtype == np.int32)
        self.assertTrue(dark_mean.dtype == np.float32)

        # Test single frame
        dark_frame, dark_mode, dark_mean = create_dark_cube(len_x=len_x, len_y=len_y, len_l=1, rand_low=rand_low, rand_high=rand_high, start=start, stop=stop, axis=axis)
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

        white_cube = create_white_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(white_cube.shape, (len_x, len_y, len_l))
        self.assertTrue(white_cube.dtype == np.int32)

        # Test single frame
        white_frame = create_white_cube(len_x=len_x, len_y=len_y, len_l=1, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(white_frame.shape, (len_x, len_y, 1))

        if DEBUG:
            plt.imshow(white_cube[:, :, int(len_l/2)])
            plt.title("White cube middle frame")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(white_cube[int(len_x/2), int(len_y/2), :])
            plt.title("White cube middle pixel spectrum")
            plt.show()

    def test_distances(self):
        # Test that distances at corners are 1 and in the center 0
        frame = np.zeros((3,3))
        distances = distances_to_center(frame)
        assert_almost_equal(distances[0][0], 1.0, decimal=16)
        assert_almost_equal(distances[0][2], 1.0, decimal=16)
        assert_almost_equal(distances[2][0], 1.0, decimal=16)
        assert_almost_equal(distances[2][2], 1.0, decimal=16)
        assert_almost_equal(distances[1][1], 0.0, decimal=16)

        # Test that only 2D arrays are accepted
        non_frame = np.zeros((2,2,2))
        with self.assertRaises(ValueError):
            distances_to_center(non_frame)
        non_frame = np.zeros((2,))
        with self.assertRaises(ValueError):
            distances_to_center(non_frame)

    def test_signal_cube(self):
        len_x = 250
        len_y = 200
        len_l = 50
        amplitude = 50000
        start = 400
        stop = 2000

        signal_cube = create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        self.assertEqual(signal_cube.shape, (len_x, len_y, len_l))
        self.assertTrue(signal_cube.dtype == np.int32)

        if DEBUG:
            plt.imshow(signal_cube[:, :, int(len_l * 0.5)])
            plt.title("Signal cube middle frame")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(signal_cube[int(len_x/2), int(len_y/2), :])
            plt.title("Signal cube middle pixel spectrum")
            plt.show()

    def test_vignetting(self):
        len_x = 250
        len_y = 200
        len_l = 50
        amplitude = 50000
        start = 400
        stop = 2000

        signal_cube = create_signal_cube(len_x=len_x, len_y=len_y, len_l=len_l, amplitude=amplitude, start=start, stop=stop)
        vignetted_cube = kinda_wignetting(signal_cube)

        if DEBUG:
            plt.imshow(vignetted_cube[:, :, int(len_l * 0.5)])
            plt.title("Vignetted signal cube middle frame")
            plt.colorbar()
            plt.show()
            plt.close()

            plt.plot(vignetted_cube[int(len_x/2), int(len_y/2), :])
            plt.title("Vignetted signal cube middle pixel spectrum")
            plt.show()

    def test_dark_correction(self):

        print("Im a test")
        return

        # Create a sample cube and dark cube
        cube = np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]])
        dark = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

        # Expected result after dark correction
        expected_result = np.array([[[9, 18, 27], [36, 45, 54]], [[63, 72, 81], [90, 99, 108]]])

        # Perform dark correction
        result = cl.dark_correction(cube, dark, axis=(0, 1))

        # Assert the result is as expected
        np.testing.assert_array_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()

"""
Generates mockup spectral data for testing purposes.
"""

import numpy as np
from scipy.stats import mode

def create_wave(n, m, wave="sine", start=400, stop=2500):
    """
    Create a sine or cosine wave from start to stop with n points, m periods and amplitude from 0 to 1.

    :param n:
        Number of points in the wave.
    :param m:
        Number of periods in the wave.
    :param wave:
        Type of wave. Either "sine" or "cosine".
    :param start:
        Start wavelength for the wave.
    :param stop:
        Stop wavelength for the wave.
    :return:
        wls, intensities: Wavelengths and intensities as numpy arrays
    """

    x = np.linspace(0, 2 * np.pi * m, n)
    wls = np.linspace(start, stop, n)
    wls = wls.astype(np.uint64)
    if wave == "sine":
        intensities = np.array(0.5 * np.sin(x) + 0.5)
    else:
        intensities = np.array(0.5 * np.cos(x) + 0.5)

    return wls, intensities


def create_lamp_spectrum(amplitude, n, start=400, stop=2500):
    """Generates a quarter of a sine wave with n points and amplitude from 0 to amplitude.

    :param amplitude:
        Amplitude of the lamp spectrum.
    :param n:
        Number of points in the lamp spectrum.
    :param start:
        Start wavelength for the lamp spectrum.
    :param stop:
        Stop wavelength for the lamp spectrum.
    :return:
        wls, intensities: Wavelengths and intensities as numpy arrays
    """

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
    """Generates a dark cube with len_l spectral points and len_x * len_y spatial points.

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
    """Applies a simple pseudo vignetting effect to the cube.

    :param cube:
        2D or 3D numpy array representing the cube.
    :return:
        2D or 3D numpy array representing the cube with vignetting applied.
    """

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

    The spectral signal is sine or cosine wave.

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

import os

import numpy as np
from spectral.io import envi as envi


def find_cube(path: str, mode: str, save_state, runtime_state):
    """Finds a cube, and if it is OK, opens it.

    :param path:
        Path to the cube. Can be path to either .hdr, .raw, .log etc.
        Correct files are inferred based on the base name of the file
        without the postfix.
    :param mode:
        Either 'cube', 'dark' or 'white'.
    """

    selected_dir_path = os.path.dirname(path)
    selected_file_name = os.path.basename(path)
    base_name = selected_file_name.rsplit(sep='.', maxsplit=1)[0]

    print(f"File: '{selected_file_name}' in '{selected_dir_path}'. Base name is '{base_name}'.")

    hdr_found = False
    raw_found = False
    reflectance_found = False

    hdr_file_name = None
    cube_file_name = None

    file_list = os.listdir(selected_dir_path)
    for file_name in file_list:
        fn_wo_postfix = file_name.rsplit(sep='.', maxsplit=1)[0]
        if fn_wo_postfix == base_name and file_name.lower().endswith(".hdr"):
            hdr_found = True
            hdr_file_name = file_name
        if fn_wo_postfix == base_name and file_name.lower().endswith(".raw"):
            raw_found = True
            cube_file_name = file_name
        if fn_wo_postfix == base_name and (file_name.lower().endswith(".dat") or file_name.lower().endswith(".img")):
            reflectance_found = True
            cube_file_name = file_name

    if hdr_found and (raw_found or reflectance_found):
        print(f"Envi cube files OK. ")
        hdr_path = os.path.join(selected_dir_path, hdr_file_name)
        raw_path = os.path.join(selected_dir_path, cube_file_name)

        if mode == 'cube':
            save_state['main_cube_hdr_path'] = hdr_path
            save_state['main_cube_data_path'] = raw_path

            # Set the flag after the cube is properly loaded
            runtime_state['cube_is_reflectance'] = reflectance_found
            if reflectance_found:
                # We'll need to put white_corrected flag also on, so that the RGB draw knows we are dealing with floats
                runtime_state['white_corrected'] = True
        elif mode == 'dark':
            save_state['dark_cube_hdr_path'] = hdr_path
            save_state['dark_cube_data_path'] = raw_path
        elif mode == 'white':
            save_state['white_cube_hdr_path'] = hdr_path
            save_state['white_cube_data_path'] = raw_path
        else:
            print(f"ERROR Unsupported mode '{mode}' for find_cube().")

        # We managed to find a proper file, so might as well set the path to memory for saving plots
        runtime_state['cube_dir_path'] = selected_dir_path
        # open_cube(hdr_path=hdr_path, data_path=raw_path, mode=mode)
        can_open = True
    else:
        print(f"Not OK. Either hdr or raw file not found from given directory.")
        can_open = False

    return can_open, hdr_path, raw_path


def open_cube(hdr_path, data_path, mode, runtime_state):
    """Opens a hyperspectral image cube.

    Sets pixel plot axis from metadata if mode == 'cube'.
    Connects mouse click handlers and updates the false color canvas.

    :param hdr_path:
        Path to the ENVI header file.
    :param data_path:
        Path to the ENVI data file.
    :param mode:
        One of 'cube', 'dark' or 'white'.
    """

    if hdr_path is None or data_path is None:
        print(f"Either HDR path or DATA path was None. Cannot open cube.")
        return

    cube_data = envi.open(file=hdr_path, image=data_path)
    img_array = cube_data.load().asarray()

    if mode == 'cube':

        runtime_state['cube_data'] = cube_data
        runtime_state['img_array'] = img_array

        # Set things up using metadata
        # TODO this should work even if the cube is not selected? Perhaps not as it doesn't make much sense.
        # clear_plot()
        # cube_meta()

    elif mode == 'dark':
        runtime_state['img_array_dark'] = img_array
    elif mode == 'white':
        runtime_state['img_array_white'] = img_array
    else:
        print(f"WARNING: Unsupported mode '{mode}' for open_cube().")



def calc_dark(runtime_state):
    """Calculates dark correction for current cube.

    Updates the false color canvas after done.
    """

    print(f"Dark calculation called...")

    if runtime_state['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    dark_spectrum = runtime_state['dark_median']
    if dark_spectrum is None:
        print(f"Cannot calculate dark because dark median is None. Select a dark cube first.")
        return

    # FIXME subtract the MEDIAN of the dark. Not the mean
    runtime_state['img_array'] = runtime_state['img_array'] - dark_spectrum
    runtime_state['img_array'] = np.clip(runtime_state['img_array'], a_min=0, a_max=None)
    runtime_state['img_array'] = runtime_state['img_array']

    runtime_state['view_mode'] = 'cube'
    runtime_state['dark_corrected'] = True


def calc_white(runtime_state):

    print(f"White calculation called...")

    if runtime_state['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    white_spectrum = runtime_state['white_spectra']
    if white_spectrum is None:
        print(f"Cannot calculate white because white spectrum is None. Select a region from the white cube first.")
        return

    white_spectrum = white_spectrum - runtime_state['dark_median']

    runtime_state['img_array'] = np.divide(runtime_state['img_array'], white_spectrum, dtype=np.float32)
    runtime_state['white_corrected'] = True

    runtime_state['view_mode'] = 'cube'


def cube_meta(text_component, runtime_state):
    """Read cube metadata and print on given UI element.

    Also sets pixel plot axis according to metadata.
    """

    cube_data = runtime_state['cube_data']

    if cube_data is None:
        print(f"Cube data not set. Returning without doing nothing.")
        return

    walength_set = False

    for key,value in cube_data.metadata.items():
        if key.lower() == 'wavelength':
            runtime_state['cube_wls'] = np.array(list(float(v) for v in value))
            # print(_VARS['cube_wls'])
            runtime_state['cube_bands'] = np.arange(start=0, stop=len(value), step=1)
            runtime_state['spectral_clip_min'] = 0
            runtime_state['spectral_clip_max'] = len(value) - 1
            # print(f"len wls: {len(_VARS['cube_wls'])}")
            # print(f"len bands: {len(_VARS['cube_bands'])}")
            walength_set = True
        elif key.lower() == 'fwhm':
            # TODO use these for variance in Gaussian for RGB?
            continue
        else:
            text_component.print(f"{key}: {value}")

    if not walength_set:
        print(f"Could not set wavelengths and bands.")
        return

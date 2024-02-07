"""

Cube Inspector is almost fully implemented in this one main file.


Some resources that might be useful:

Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/
Github project PySimpleGUI: https://github.com/PySimpleGUI/PySimpleGUI

Refreshing plot: https://gist.github.com/KenoLeon/e913de9e1fe690ebe287e6d1e54e3b97
"""

# import diffbox
#
# diffbox.thing()
#
# exit(0)

import os
import math
import logging

import PySimpleGUI as sg
import spectral.io.envi as envi
import spectral as spy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import toml_handling as TH

# TODO Incorporate dark and white selection logic with buttons.
# TODO Show corresponding wl for RGB band selection. In plot or as a text box?
# TODO Gaussian or mean RGB.

# Set Matplotlib to use TKinter backend apparently?
matplotlib.use("TkAgg")

# Set Spectral Python library to support non-lowercase file names
spy.settings.envi_support_nonlowercase_params = True

# Set a nice color theme
sg.theme('dark grey 9')

# State dict save directory and filename
state_dir_path = os.getcwd()
state_file_name = "ci_state"


# GUI entity keys ###################

# Browse file buttons
guiek_cube_file_browse = "-CUBE BROWSE-"
guiek_dark_file_browse = "-DARK BROWSE-"
guiek_white_file_browse = "-WHITE BROWSE-"

# Show selected filename
guiek_cube_show_filename = "-CUBE SHOW FILENAME-"
guiek_dark_show_filename = "-DARK SHOW FILENAME-"
guiek_white_show_filename = "-WHITE SHOW FILENAME-"

# For show buttons
guiek_cube_show_button = "-CUBE SHOW BUTTON-"
guiek_dark_show_button = "-DARK SHOW BUTTON-"
guiek_white_show_button = "-WHITE SHOW BUTTON-"

# White reference selection buttons
guiek_white_select_region = "-WHITE SELECT REGION-"
guiek_white_select_whole = "-WHITE SELECT WHOLE-"

# RGB selection
guiek_r_input = "-R-"
guiek_g_input = "-G-"
guiek_b_input = "-B-"
guiek_r_wl_text = "-R WAVELENGTH-"
guiek_g_wl_text = "-G WAVELENGTH-"
guiek_b_wl_text = "-B WAVELENGTH-"
guiek_rgb_update_button = "-RGB UPDATE BUTTON-"

# Correction calculation buttons
guiek_calc_dark = "-CALCULATE DARK-"
guiek_calc_white = "-CALCULATE WHITE-"

# Canvases
guiek_cube_false_color = "-FALSE COLOR-"
guiek_pixel_plot_canvas = "-PX PLOT CANVAS-"

# Cube metadata
guiek_cube_meta_text = "-CUBE META-"

# UI console
guiek_console_output = "-CONSOLE-"

# Save cube button
guiek_save_cube = "-SAVE CUBE-"

# Pixel plot component handles
guiek_clear_button = "-CLEAR-"
guiek_spectral_min_input = "-SPECTRAL MIN INPUT-"
guiek_spectral_max_input = "-SPECTRAL MAX INPUT-"
guiek_spectal_clip_min_wl_text = "-SPECTRAL CLIP MIN TEXT WL-"
guiek_spectal_clip_max_wl_text = "-SPECTRAL CLIP MAX TEXT WL-"
guiek_spectral_clip_button = "-SPECTRAL CLIP-"

# Pixel plot
fig_px_plot = plt.figure(figsize=(5, 4), dpi=100)
ax_px_plot = fig_px_plot.add_subplot(111)

# False color imshow
fig_false_color = plt.figure(figsize=(5, 4), dpi=100)
ax_false_color = fig_false_color.add_subplot(111)


# Layout stuff #############

cube_meta_column = [
    [
        sg.Text("Cube metadata"),
    ],
    [
        sg.Multiline(size=(50, 15), key=guiek_cube_meta_text),
    ],
]


cube_column = [
    [
        sg.Text("Cube"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_cube_show_filename, disabled=True, text_color='black'), # always disabled
        sg.FileBrowse(key=guiek_cube_file_browse, target=guiek_cube_file_browse, enable_events=True,
                      tooltip="Select a hyperspectral ENVI format hypercube to show. It suffices to pick either a header or \n"
                              "data file, the directory is searched for associated files automatically. The data can \n"
                              "be as raw data (radiance, digital number, etc.) or precomputed reflectance cube with \n"
                              "dark and white corrections."), # always enabled
        sg.Button('Show', enable_events=True, key=guiek_cube_show_button, disabled=True,
                  tooltip="Shows selected cube. This is used to swap back and forth with dark, white and main cubes if applicable, i.e., \n"
                          "when not dealing with precomputed reflectance cubes."),
    ],
    [
        sg.Text("Dark"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_dark_show_filename, disabled=True, text_color='black'),# always disabled
        sg.FileBrowse(key=guiek_dark_file_browse, target=guiek_dark_file_browse, enable_events=True, disabled=True,
                      tooltip="Select a dark cube to be used for dark current correction. Median of the cube is calculated \n"
                              "automatically after selection, so give it some time."),
        sg.Button('Show', enable_events=True, key=guiek_dark_show_button, disabled=True,
                  tooltip="Shows selected DARK cube. This is used to swap back and forth with dark, white and main cubes if applicable, i.e., \n"
                          "when not dealing with precomputed reflectance cubes."
                  ),
        sg.Button("Calculate", k=guiek_calc_dark, disabled=True,
                  tooltip="Subtracts dark current from the main cube. \n Dark current is the median of the whole dark cube over scan lines."),
    ],
    [
        sg.Text("White"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_white_show_filename, disabled=True, text_color='black', background_color='grey'),# always disabled
        sg.FileBrowse(key=guiek_white_file_browse, target=guiek_white_file_browse, enable_events=True, disabled=True,
                      tooltip="Select white reference cube for white correction. If your white reference is in your main data cube, \n"
                              "you can select it again and use the 'Select region' button to select the pixels you want to use.\n"),
        sg.Button('Show', enable_events=True, key=guiek_white_show_button, disabled=True,
                  tooltip="Shows selected WHITE cube. This is used to swap back and forth with dark, white and main cubes if applicable, i.e., \n"
                          "when not dealing with precomputed reflectance cubes."
                  ),
        sg.Button("Select region", k=guiek_white_select_region, disabled=True,
                  tooltip="Select a region of the white cube to be used as a white reference."),
        sg.Button("Select whole", k=guiek_white_select_whole, disabled=True,
                  tooltip="Select the whole white cube to be used as a white reference."),
        sg.Button("Calculate", k=guiek_calc_white, disabled=True,
                  tooltip="Calculate white correction. The each pixel (spectrum) will be divided by the white reference spectrum, \n"
                          "which is the band-wise mean of all the selected white reference area."),
    ],
    [
        sg.Canvas(key=guiek_cube_false_color),
    ],
    [
        sg.Text("R"),
        sg.In(size=(5, 1), key=guiek_r_input,
              tooltip="Band used for the red channel in the false color representation of the cube."),
        sg.Text("---.-- nm", key=guiek_r_wl_text),
        sg.Text("G"),
        sg.In(size=(5, 1), key=guiek_g_input,
              tooltip="Band used for the green channel in the false color representation of the cube."),
        sg.Text("---.-- nm", key=guiek_g_wl_text),
        sg.Text("B"),
        sg.In(size=(5, 1), key=guiek_b_input,
              tooltip="Band used for the blue channel in the false color representation of the cube."),
        sg.Text("---.-- nm", key=guiek_b_wl_text),
        sg.Button('Update', enable_events=True, key=guiek_rgb_update_button,
                  tooltip="Update false color image and pixel plot to represent the selected bands.")
    ],
]

pixel_plot_column = [
    [sg.Canvas(key=guiek_pixel_plot_canvas)],
    [
        sg.Button("Clear", key=guiek_clear_button),
        sg.Text("Min: "),
        sg.In(size=(5,1), key=guiek_spectral_min_input, tooltip="Spectral range low clip."),
        sg.Text("---.-- nm", key=guiek_spectal_clip_min_wl_text),
        sg.Text("Max: "),
        sg.In(size=(5,1), key=guiek_spectral_max_input, tooltip="Spectral range high clip."),
        sg.Text("---.-- nm", key=guiek_spectal_clip_max_wl_text),
        sg.Button("Clip", key=guiek_spectral_clip_button, enable_events=True)
    ]
]

layout = [
    [
        sg.Column(cube_meta_column),
        sg.VSeperator(),
        sg.Column(cube_column),
        sg.VSeperator(),
        sg.Column(pixel_plot_column),
    ],
    [sg.HSeparator()],
    [
        sg.Multiline(size=(120, 15), reroute_stdout=True, k=guiek_console_output, autoscroll=True, horizontal_scroll=True),
        sg.Button("Save", key=guiek_save_cube, enable_events=True, disabled=True,
                  tooltip="Saves the main cube as reflectance cube (with .dat extension). Only available \n"
                          "after dark and white corrections are calculated. Not available for precomputed \n "
                          "reflectance cubes.")
    ],
]

window = sg.Window("Cube Inspector", layout=layout, margins=(100,100), finalize=True)
window[guiek_console_output].Widget.configure(wrap='none')


def draw_figure(canvas, figure):
    """Helper function to draw things on canvas."""

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


# Keep most of the global stuff in this single dictionary for later access
_RUNTIME = {
    'window': window,
    'fig_agg_px_plot': draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot),
    'sec_axes_px_plot': None,
    'fig_agg_false_color': draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color),
    'pltFig': False,

    'cube_is_reflectance': False, # If loaded cube is already reflectance we can disable all calculation stuff
    'cube_data': None,
    'img_array': None,
    'img_array_dark': None,
    'img_array_white': None,
    'white_corrected': False,
    'dark_corrected': False,
    'cube_wls': None,
    'cube_bands': None,
    'rect_x_0': None,
    'rect_y_0': None,
    'rect_x_1': None,
    'rect_y_1': None,

    'rectangle_handles': [],
    'dot_handles': [],
    'rgb_handles': [],

    'selecting_white': False,

    'view_mode': None, # 'cube', 'dark', 'white' or None

    'white_spectra': None,
    'dark_median': None,

    'mouse_handlers_connected': False,

    'band_B': '', # string
    'band_G': '', # string
    'band_R': '', # string

    'spectral_clip_min': 0,
    'spectral_clip_max': 0,

    'plots': [],
}


_STATE = {
    'main_cube_hdr_path': None,
    'dark_cube_hdr_path': None,
    'white_cube_hdr_path': None,
    'main_cube_data_path': None,
    'dark_cube_data_path': None,
    'white_cube_data_path': None,

    'band_B': 0,
    'band_G': 0,
    'band_R': 0,
}

_SETTINGS = {
    'drag_threshold': 5
}


def mouse_click_event(eventi):
    """Handle mouse click event (button 1 down)."""

    # Store data in case there will be a drag
    _RUNTIME['rect_x_0'] = eventi.xdata
    _RUNTIME['rect_y_0'] = eventi.ydata


def mouse_release_event(eventi):
    """Handles Matplotlib mouse button release event.

    Note that you cannot raise PySimpleGUI popups in here as it will block the event loop.

    :param eventi:
        Matplotlib mouse button release event.
    """

    # Left click for right-handed mouse.
    if eventi.button == 1:
        x0 = _RUNTIME['rect_x_0']
        y0 = _RUNTIME['rect_y_0']
        y = eventi.ydata
        x = eventi.xdata

        drag_treshold = _SETTINGS['drag_threshold']

        # We have a drag if we have previously set (and not cleared) x0 and y0 and the release position is far enough away.
        if (x0 is not None or y0 is not None) and (math.fabs(x0 - x) > drag_treshold or math.fabs(y0 - y) > drag_treshold):

            # print(f"Mouse dragged from ({x0}, {y0}) to ({x}, {y})")
            drag_start_x = int(min(x, x0))
            drag_start_y = int(min(y, y0))
            drag_end_x = int(max(x, x0))
            drag_end_y = int(max(y, y0))

            print(f"Mouse drag from ({int(drag_start_x)},{int(drag_start_y)}) to ({int(drag_end_x)},{int(drag_end_y)}).")

            rows = list(np.arange(start=drag_start_x, stop=drag_end_x, step=1))
            cols = list(np.arange(start=drag_start_y, stop=drag_end_y, step=1))

            if _RUNTIME['view_mode'] == 'cube':
                sub_image = _RUNTIME['img_array'][cols][:, rows]
            elif _RUNTIME['view_mode'] == 'dark':
                sub_image = _RUNTIME['img_array_dark'][cols][:, rows]
            elif _RUNTIME['view_mode'] == 'white':
                sub_image = _RUNTIME['img_array_white'][cols][:, rows]
            else:
                print(f"WARNING: View mode '{_RUNTIME['view_mode']}' not supported.")

            sub_mean = np.mean(sub_image, axis=(0,1))
            sub_std = np.std(sub_image, axis=(0,1))

            update_px_rgb_lines()
            update_px_plot(spectrum=sub_mean, std=sub_std, x0=drag_start_x, y0=drag_start_y, x1=drag_end_x, y1=drag_end_y)

            if _RUNTIME['selecting_white']:
                _RUNTIME['white_spectra'] = sub_mean
                _RUNTIME['selecting_white'] = False
                print(f"White spectrum saved.")

        else:
            # Else it was just a click and we can reset x0 and y0
            _RUNTIME['rect_x_0'] = None
            _RUNTIME['rect_y_0'] = None
            pixel = _RUNTIME['img_array'][int(y), int(x)]
            print(f"Mouse click at ({int(x)},{int(y)}).")
            update_px_rgb_lines()
            update_px_plot(spectrum=pixel, x0=x, y0=y)

        update_UI_component_state()


def infer_runtime_spectral_clip(min_value: str, max_value: str):
    """

    :param min_value:
    :param max_value:
    :return:
    :raises:
        ValueError if min > max.
    """

    if min_value is None or len(min_value) < 0 or max_value is None or len(max_value) < 0:
        raise ValueError(f"Either min or max value was not provided.")

    min_int = int(min_value)
    max_int = int(max_value)

    if min_int < 0 or max_int < 0:
        raise ValueError(f"Negative values not allowed. Min was ({min_int}) and max ({max_int}).")

    if min_int > max_int:
        raise ValueError(f"Clipping min ({min_int}) greater than max ({max_int}).")

    if _RUNTIME['cube_data'] is None:
        raise RuntimeError(f"No cube selected: I will not set clipping for you.")

    bands = _RUNTIME['cube_bands']
    actual_min = np.clip(min_int, bands[0], bands[-1])
    actual_max = np.clip(max_int, bands[0], bands[-1])
    _RUNTIME['spectral_clip_min'] = actual_min
    _RUNTIME['spectral_clip_max'] = actual_max


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


def update_px_rgb_lines():
    """Update the vertical line positions for pixel plot to the _RUNTIME.

    Additionally, updates the wavelength textblocks of selected bands.

    NOTE
    You should call update_px_plot() after calling this for the actual update on screen.
    """

    for handle in _RUNTIME['rgb_handles']:
        handle.remove()

    _RUNTIME['rgb_handles'] = []
    band_keys = ['band_R', 'band_G', 'band_B']
    line_colors = ['red', 'green', 'blue']

    for i in range(3):
        should_fill, value = infer_runtime_RGB_value(_RUNTIME[band_keys[i]])
        if not should_fill:
            _RUNTIME['rgb_handles'].append(ax_px_plot.axvline(x=value, color=line_colors[i]))

    update_band_wl_textblocks()


def update_band_wl_textblocks():

    wl_textblock_keys = [guiek_r_wl_text, guiek_g_wl_text, guiek_b_wl_text]
    band_keys = ['band_R', 'band_G', 'band_B']
    for i in range(3):
        should_fill, value = infer_runtime_RGB_value(_RUNTIME[band_keys[i]])
        if not should_fill and _RUNTIME['cube_wls'] is not None:
            _RUNTIME['window'][wl_textblock_keys[i]].update(f"{_RUNTIME['cube_wls'][value]:.2f} nm")
        elif should_fill:
            _RUNTIME['window'][wl_textblock_keys[i]].update(f"---.-- nm")


def update_spectral_clip_wl_text():
    guiek_spectal_clip_min_wl_text
    _RUNTIME['window'][guiek_spectal_clip_min_wl_text].update(f"{_RUNTIME['cube_wls'][_RUNTIME['spectral_clip_min']]:.2f} nm")
    _RUNTIME['window'][guiek_spectal_clip_max_wl_text].update(f"{_RUNTIME['cube_wls'][_RUNTIME['spectral_clip_max']]:.2f} nm")


def update_px_plot(spectrum: np.array=None, std: np.array=None, x0=None, y0=None, x1=None, y1=None):
    """Update the pixel plot canvas when clicking or dragging over false color RGB canvas.

    :param spectrum:
        Spectrum to plot.
    :param std:
        If dragging, this is the standard deviation of the selected area that is
        drawn as a shadow over spectrum.
    :param x0:
        If clicking, click x location. If dragging, drag start x location.
    :param y0:
        If clicking, click y location. If dragging, drag start y location.
    :param x1:
        If dragging, drag end x location. Ignored if clicking.
    :param y1:
        If dragging, drag end y location. Ignored if clicking.
    """

    # Draw new plot and refersh canvas
    _RUNTIME['fig_agg_px_plot'].get_tk_widget().forget()
    ax_px_plot.set_xlim(_RUNTIME['spectral_clip_min'], _RUNTIME['spectral_clip_max'])
    update_spectral_clip_wl_text()

    # Plot the plot and save it. Update
    if spectrum is not None:
        ax_px_plot.plot(spectrum)
        _RUNTIME['plots'].append(spectrum)
        new_plot_max = 0
        for plot in _RUNTIME['plots']:
            plot_max = np.max(plot)
            if plot_max > new_plot_max:
                new_plot_max = plot_max

        # print(f"Setting ylim to {_RUNTIME['plot_max']}")
        # Set ylim a little higher than the max value of any of the plots
        ax_px_plot.set_ylim(0, new_plot_max * 1.05)

    bands = _RUNTIME['cube_bands']
    wls = _RUNTIME['cube_wls']

    if bands is not None and wls is not None and _RUNTIME['sec_axes_px_plot'] is None:

        # Hack to print into light form for HyperBlend
        # s_max = np.max(spectrum)
        # for i,wl in enumerate(wls):
        #     print(f"{wl:.4} {spectrum[i]/s_max:.9}")

        ax_px_plot.set_xlabel('Band')

        def forward(x):
            return np.interp(x, bands, wls)

        def inverse(x):
            return np.interp(x, wls, bands)

        # FIXME the wavelength axis is not perfect as two figures are shown at the extremes
        secax = ax_px_plot.secondary_xaxis('top', functions=(forward, inverse))
        secax.set_xlabel(r"Wavelength [$nm$]")
        _RUNTIME['sec_axes_px_plot'] = secax

    if std is not None:
        ax_px_plot.fill_between(_RUNTIME['cube_bands'], spectrum - (std / 2), spectrum + (std / 2), alpha=0.2)

    _RUNTIME['fig_agg_px_plot'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

    # Draw rectangle over RGB canvas and refresh
    if std is not None and x0 is not None and y0 is not None and x1 is not None and y1 is not None:
        width = math.fabs(x0 - x1)
        height = math.fabs(y0 - y1)
        _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
        handle = ax_false_color.add_patch(Rectangle((x0, y0), width=width, height=height, fill=False, edgecolor='gray'))
        _RUNTIME['rectangle_handles'].append(handle)
        _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

    # Draw click location as a dot
    elif x0 is not None and y0 is not None:
        _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
        handle = ax_false_color.scatter(int(x0), int(y0))
        _RUNTIME['dot_handles'].append(handle)
        _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def update_false_color_canvas():
    """Updates the false color canvas based on _RUNTIME globals.

    Uses user-defined bands for RGB reconstruction. Pixel values are automatically
    scaled for the whole possible range.

    Check if RGB exists in _RUNTIME and use that
    _RUNTIME RGB has to be loaded from _STATE
    If not, get RGB from metadata.
    """

    if _RUNTIME['cube_data'] is None:
        print(f"Cube not set. Nothing to clear.")
        return

    possibly_R_str = str(_RUNTIME['band_R'])
    possibly_G_str = str(_RUNTIME['band_G'])
    possibly_B_str = str(_RUNTIME['band_B'])
    meta_bands = [int(band) - 1 for band in _RUNTIME['cube_data'].metadata['default bands']]

    if len(possibly_R_str) == 0: # empty string
        possibly_R_str = str(meta_bands[0])
        _RUNTIME['band_R'] = possibly_R_str
    if len(possibly_G_str) == 0: # empty string
        possibly_G_str = str(meta_bands[1])
        _RUNTIME['band_G'] = possibly_G_str
    if len(possibly_B_str) == 0: # empty string
        possibly_B_str = str(meta_bands[2])
        _RUNTIME['band_B'] = possibly_B_str

    view_mode = _RUNTIME['view_mode']

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

    if view_mode == 'cube' and not _RUNTIME['selecting_white']:

        if _RUNTIME['img_array'] is None:
            print(f"Image array None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(_RUNTIME['img_array'], possibly_R_str, possibly_G_str, possibly_B_str)

    elif view_mode == 'dark':

        if _RUNTIME['img_array_dark'] is None:
            print(f"Image array for dark is None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(_RUNTIME['img_array_dark'], possibly_R_str, possibly_G_str, possibly_B_str)

    elif view_mode == 'white' or _RUNTIME['selecting_white']:

        if _RUNTIME['img_array_white'] is None:
            print(f"Image array for white is None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(_RUNTIME['img_array_white'], possibly_R_str, possibly_G_str, possibly_B_str)
    else:
        print(f"WARNING: unknown view mode '{view_mode}' and/or selection combination selecting "
              f"white={_RUNTIME['selecting_white']}.")
        return

    _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
    # Clear axis object because if the next image is of different size,
    # it will break pixel indexing for mouse selection
    ax_false_color.cla()
    ax_false_color.imshow(false_color_rgb)
    ax_false_color.set_xlabel('Samples')
    ax_false_color.set_ylabel('Lines')
    _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def cube_meta():
    """Read cube metadata and print on UI element.

    Also sets pixel plot axis according to metadata.
    """

    cube_data = _RUNTIME['cube_data']

    if cube_data is None:
        print(f"Cube data not set. Returning without doing nothing.")
        return

    walength_set = False

    for key,value in cube_data.metadata.items():
        if key.lower() == 'wavelength':
            _RUNTIME['cube_wls'] = np.array(list(float(v) for v in value))
            # print(_VARS['cube_wls'])
            _RUNTIME['cube_bands'] = np.arange(start=0, stop=len(value), step=1)
            _RUNTIME['spectral_clip_min'] = 0
            _RUNTIME['spectral_clip_max'] = len(value) - 1
            update_spectral_clip_wl_text()
            # print(f"len wls: {len(_VARS['cube_wls'])}")
            # print(f"len bands: {len(_VARS['cube_bands'])}")
            walength_set = True
        elif key.lower() == 'fwhm':
            # TODO use these for variance in Gaussian for RGB?
            continue
        else:
            window[guiek_cube_meta_text].print(f"{key}: {value}")

    if not walength_set:
        print(f"Could not set wavelengths and bands.")
        return


def find_cube(path: str, mode: str):
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
            _STATE['main_cube_hdr_path'] = hdr_path
            _STATE['main_cube_data_path'] = raw_path

            # Set the flag after the cube is properly loaded
            _RUNTIME['cube_is_reflectance'] = reflectance_found
            if reflectance_found:
                # We'll need to put white_corrected flag also on, so that the RGB draw knows we are dealing with floats
                _RUNTIME['white_corrected'] = True
        elif mode == 'dark':
            _STATE['dark_cube_hdr_path'] = hdr_path
            _STATE['dark_cube_data_path'] = raw_path
        elif mode == 'white':
            _STATE['white_cube_hdr_path'] = hdr_path
            _STATE['white_cube_data_path'] = raw_path
        else:
            print(f"ERROR Unsupported mode '{mode}' for find_cube().")

        open_cube(hdr_path=hdr_path, data_path=raw_path, mode=mode)
    else:
        print(f"Not OK. Either hdr or raw file not found from given directory.")


def open_cube(hdr_path, data_path, mode):
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

        _RUNTIME['cube_data'] = cube_data
        _RUNTIME['img_array'] = img_array

        # Set things up using metadata
        # TODO this should work even if the cube is not selected? Perhaps not as it doesn't make much sense.
        clear_plot()
        cube_meta()

    elif mode == 'dark':
        _RUNTIME['img_array_dark'] = img_array
    elif mode == 'white':
        _RUNTIME['img_array_white'] = img_array
    else:
        print(f"WARNING: Unsupported mode '{mode}' for open_cube().")

    # Connect mouse click
    if not _RUNTIME['mouse_handlers_connected']:
        cid_press = fig_false_color.canvas.mpl_connect('button_press_event', mouse_click_event)
        cid_release = fig_false_color.canvas.mpl_connect('button_release_event', mouse_release_event)
        _RUNTIME['mouse_handlers_connected'] = True

    # Draw cube to canvas
    update_false_color_canvas()
    update_band_wl_textblocks()


def calc_dark():
    """Calculates dark correction for current cube.

    Updates the false color canvas after done.
    """

    print(f"Dark calculation called...")

    if _RUNTIME['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    dark_spectrum = _RUNTIME['dark_median']
    if dark_spectrum is None:
        print(f"Cannot calculate dark because dark median is None. Select a dark cube first.")
        return

    # FIXME subtract the MEDIAN of the dark. Not the mean
    _RUNTIME['img_array'] = _RUNTIME['img_array'] - dark_spectrum
    _RUNTIME['img_array'] = np.clip(_RUNTIME['img_array'], a_min=0, a_max=None)
    _RUNTIME['img_array'] = _RUNTIME['img_array']

    _RUNTIME['view_mode'] = 'cube'
    _RUNTIME['dark_corrected'] = True
    update_false_color_canvas()


def calc_white():

    print(f"White calculation called...")

    if _RUNTIME['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    white_spectrum = _RUNTIME['white_spectra']
    if white_spectrum is None:
        print(f"Cannot calculate white because white spectrum is None. Select a region from the white cube first.")
        return

    _RUNTIME['img_array'] = np.divide(_RUNTIME['img_array'], white_spectrum, dtype=np.float32)
    _RUNTIME['white_corrected'] = True

    _RUNTIME['view_mode'] = 'cube'
    update_false_color_canvas()


def clear_plot():
    """Clear pixel plot.

    TODO maybe make more general to clear any pyplot axis?
    """

    print(f"Clearing pixel plot")
    _RUNTIME['fig_agg_px_plot'].get_tk_widget().forget()
    ax_px_plot.clear()
    _RUNTIME['fig_agg_px_plot'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

    print(f"Clearing false color RGB")
    # Remove rectangles and other Artists
    # while ax_false_color.patches:
    #     ax_false_color.patches[0].remove()
    for handle in _RUNTIME['rectangle_handles']:
        handle.remove()
    for handle in _RUNTIME['dot_handles']:
        handle.remove()
    for handle in _RUNTIME['rgb_handles']:
        handle.remove()

    _RUNTIME['rectangle_handles'] = []
    _RUNTIME['dot_handles'] = []
    _RUNTIME['rgb_handles'] = []
    _RUNTIME['sec_axes_px_plot'] = None
    _RUNTIME['plots'] = []
    update_false_color_canvas()


def state_save():
    """Save the _STATE dict to disk.
    """

    _STATE['band_R'] = str(_RUNTIME['band_R'])
    _STATE['band_G'] = str(_RUNTIME['band_G'])
    _STATE['band_B'] = str(_RUNTIME['band_B'])

    TH.write_dict_as_toml(dictionary=_STATE, directory=state_dir_path, filename=state_file_name)
    print(f"State saved")


def state_load():
    """Load state from disk and sets _STATE global dict.

    Also sets RGB bands to _RUNTIME global dict.
    """

    state = TH.read_toml_as_dict(directory=state_dir_path, filename=state_file_name)
    for key, value in state.items():
        _STATE[key] = value

    _RUNTIME['band_R'] = str(_STATE['band_R'])
    _RUNTIME['band_G'] = str(_STATE['band_G'])
    _RUNTIME['band_B'] = str(_STATE['band_B'])

    print(f"Previous state loaded.")


def update_UI_component_state():
    """Updates the disabled state of all UI buttons and RGB inboxes."""

    # First, disable all
    window[guiek_cube_show_button].update(disabled=True)

    window[guiek_dark_file_browse].update(disabled=True)
    window[guiek_dark_show_button].update(disabled=True)

    window[guiek_white_file_browse].update(disabled=True)
    window[guiek_white_show_button].update(disabled=True)
    window[guiek_white_select_region].update(disabled=True)
    window[guiek_white_select_whole].update(disabled=True)
    window[guiek_calc_white].update(disabled=True)

    window[guiek_save_cube].update(disabled=True)

    if _RUNTIME['img_array'] is not None:
        window[guiek_cube_show_button].update(disabled=False)

        if not _RUNTIME['cube_is_reflectance']:
            window[guiek_dark_file_browse].update(disabled=False)
            window[guiek_white_file_browse].update(disabled=False)

    # We only need calculation stuff is loaded cube is not already reflectance
    if not _RUNTIME['cube_is_reflectance']:

        if _RUNTIME['img_array_dark'] is not None:
            window[guiek_dark_show_button].update(disabled=False)
            if not _RUNTIME['dark_corrected'] and _RUNTIME['dark_median'] is not None:
                window[guiek_calc_dark].update(disabled=False)
            else:
                window[guiek_calc_dark].update(disabled=True)
        if _RUNTIME['img_array_white'] is not None:
            window[guiek_white_show_button].update(disabled=False)
            window[guiek_white_select_region].update(disabled=False)
            window[guiek_white_select_whole].update(disabled=False)
            if not _RUNTIME['white_corrected'] and _RUNTIME['white_spectra'] is not None:# and _RUNTIME['dark_corrected']:
                window[guiek_calc_white].update(disabled=False)
            else:
                window[guiek_calc_white].update(disabled=True)

        if _RUNTIME['white_corrected'] or _RUNTIME['dark_corrected']:
            window[guiek_save_cube].update(disabled=False)

    # Update the RGB inboxes as well
    window[guiek_r_input].update(str(_RUNTIME['band_R']))
    window[guiek_g_input].update(str(_RUNTIME['band_G']))
    window[guiek_b_input].update(str(_RUNTIME['band_B']))

    window[guiek_spectral_min_input].update(str(_RUNTIME['spectral_clip_min']))
    window[guiek_spectral_max_input].update(str(_RUNTIME['spectral_clip_max']))


def get_base_name_wo_postfix(path: str) -> str:
    """Returns the file name in given path without the postfix such as .hdr."""

    base_name = os.path.basename(path).rsplit('.', maxsplit=1)[0]
    return base_name


def restore_from_previous_session():
    """Partially restores UI to the state it was when last closed.

    Loads all the cubes but does not do any corrections.
    """

    print(f"Trying to restore state from previous session.")

    if _STATE['main_cube_hdr_path'] is not None:
        path = _STATE['main_cube_hdr_path']
        window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_cube_file_selected(path)
    if _STATE['dark_cube_hdr_path'] is not None:
        path = _STATE['dark_cube_hdr_path']
        window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_dark_file_selected(path)
    if _STATE['white_cube_hdr_path'] is not None:
        path = _STATE['white_cube_hdr_path']
        window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_white_file_selected(path)

    _RUNTIME['view_mode'] = 'cube'
    update_false_color_canvas()


def handle_cube_file_selected(file_path:str):

    # Reset in case a new cube is selected
    _RUNTIME['selecting_white'] = False
    _RUNTIME['view_mode'] = 'cube'

    _RUNTIME['dark_median'] = None
    _RUNTIME['img_array_dark'] = None
    _RUNTIME['dark_corrected'] = False

    _RUNTIME['white_spectra'] = None
    _RUNTIME['img_array_white'] = None
    _RUNTIME['white_corrected'] = False

    find_cube(file_path, mode='cube')
    update_false_color_canvas()
    update_UI_component_state()


def handle_dark_file_selected(file_path: str):

    _RUNTIME['view_mode'] = 'dark'

    find_cube(file_path, mode='dark')
    # image_array_dark should now have a value
    dark_cube = _RUNTIME['img_array_dark']
    print(f"Dark cube set. Calculating median of the scan lines. WAIT a bit, please.")

    # Scan lines are on axis=0
    med = np.median(dark_cube, axis=0)
    # print(f"DEBUG: dark median shape: {med.shape}")
    _RUNTIME['dark_median'] = med

    print(f"Dark median saved.")
    update_false_color_canvas()
    update_UI_component_state()


def handle_white_file_selected(file_path: str):
    _RUNTIME['selecting_white'] = True
    _RUNTIME['view_mode'] = 'white'
    find_cube(file_path, mode='white')
    _RUNTIME['selecting_white'] = False

    print(f"White cube set.")

    update_false_color_canvas()
    update_UI_component_state()


def save_reflectance_cube():
    basepath = str(_STATE['main_cube_hdr_path']).rsplit('.', maxsplit=1)[0]
    if _RUNTIME['white_corrected']:
        save_hdr_path = f"{basepath}_CI_reflectance.hdr"
    elif _RUNTIME['dark_corrected']:
        save_hdr_path = f"{basepath}_CI_darkcorrected.hdr"
    print(f"Trying to save cube to '{save_hdr_path}'.")
    try:
        spy.envi.save_image(hdr_file=save_hdr_path, image=_RUNTIME['img_array'], dtype=np.float32, ext='.dat', metadata=_RUNTIME['cube_data'].metadata)
        print(f"Cube saved.")

    except envi.EnviException as e:
        answer = sg.PopupYesNo("Cube already exists with that file name. \n Do you want to override it?")
        if answer == 'Yes':
            spy.envi.save_image(hdr_file=save_hdr_path, image=_RUNTIME['img_array'], dtype=np.float32, ext='.dat',
                                metadata=_RUNTIME['cube_data'].metadata, force=True)
            print(f"Cube saved. Old cube was overriden.")
        else:
            print(f"User cancelled saving.")



def main():
    answer = sg.popup_yes_no("Wanna load previous session?\n\n"
                             "Be patient if you select yes: it will take some time before the UI updates.")
    if answer == 'Yes':

        try:
            state_load()
            print(f"State loaded")
        except FileNotFoundError as e:
            print(f"No previous state found.")

        restore_from_previous_session()

    # Infinite GUI loop
    while True:
        event, values = window.read()

        # Clutters console but can be useful for debugging.
        # print(f"event {event}, values: {values}")

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == guiek_clear_button:
            clear_plot()

        if event == guiek_spectral_clip_button:
            try:
                infer_runtime_spectral_clip(values[guiek_spectral_min_input], values[guiek_spectral_max_input])
                update_px_plot()
            except Exception as e:
                print(f"Could not set clip value: \n {e}")

        if event == guiek_cube_file_browse:
            window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(values[guiek_cube_file_browse]))
            handle_cube_file_selected(values[guiek_cube_file_browse])

        if event == guiek_dark_file_browse:
            window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(values[guiek_dark_file_browse]))
            handle_dark_file_selected(values[guiek_dark_file_browse])

        if event == guiek_white_file_browse:
            window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(values[guiek_white_file_browse]))
            handle_white_file_selected(values[guiek_white_file_browse])

        if event == guiek_cube_show_button:
            _RUNTIME['view_mode'] = 'cube'
            update_false_color_canvas()

        if event == guiek_dark_show_button:
            _RUNTIME['view_mode'] = 'dark'
            update_false_color_canvas()

        if event == guiek_white_show_button:
            _RUNTIME['view_mode'] = 'white'
            update_false_color_canvas()

        if event == guiek_white_select_region:
            _RUNTIME['selecting_white'] = True
            print(f"White region selection is now on. Drag across the image to select an "
                  f"area which will be used as a white reference.")

        if event == guiek_white_select_whole:
            white_array = _RUNTIME['img_array_white']
            white_mean = np.mean(white_array, axis=(0,1))
            _RUNTIME['white_spectra'] = white_mean
            print(f"White reference spectra set. You can now use the Calculate button to "
                  f"calculate reflectance.")

        if event == guiek_calc_dark:
            calc_dark()

        if event == guiek_calc_white:
            calc_white()

        if event == guiek_save_cube:
            save_reflectance_cube()

        if event == guiek_rgb_update_button:
            try:
                # Just try casting before continuing
                _, _ = infer_runtime_RGB_value(values[guiek_r_input])
                _, _ = infer_runtime_RGB_value(values[guiek_g_input])
                _, _ = infer_runtime_RGB_value(values[guiek_b_input])
                _RUNTIME['band_R'] = values[guiek_r_input] #int(values[guiek_r_input])
                _RUNTIME['band_G'] = values[guiek_g_input] #int(values[guiek_g_input])
                _RUNTIME['band_B'] = values[guiek_b_input] #int(values[guiek_b_input])
                update_px_rgb_lines()
                update_px_plot()
                update_false_color_canvas()
            except ValueError as ve:
                print(f"WARNING: Failed casting band to an integer. False color image not updated.")

        # Update UI after every event is handled.
        update_UI_component_state()

    state_save()
    window.close()


if __name__ == '__main__':

    main()
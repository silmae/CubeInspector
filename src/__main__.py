"""

Main GUI loop and other stuff needed for that.

Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/
Github project PySimpleGUI: https://github.com/PySimpleGUI/PySimpleGUI

Refreshing plot: https://gist.github.com/KenoLeon/e913de9e1fe690ebe287e6d1e54e3b97
"""

import os
import math

import PySimpleGUI as sg
import spectral.io.envi as envi
import spectral as spy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator
import toml_handling as TH

matplotlib.use("TkAgg")
spy.settings.envi_support_nonlowercase_params = True

# Set a nice color theme
sg.theme('dark grey 9')


def draw_figure(canvas, figure):
    """Helper function to draw things on canvas."""

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def mouse_click_event(eventi):
    """Handle mouse click event."""

    print('x: {} and y: {}'.format(eventi.xdata, eventi.ydata))

    # TODO inverted x and y coordinates. See if this is actually necessary.
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

            update_px_plot(spectrum=sub_mean, std=sub_std, x0=drag_start_x, y0=drag_start_y, x1=drag_end_x, y1=drag_end_y)

            if _RUNTIME['selecting_white']:
                _RUNTIME['white_spectra'] = sub_mean
                _RUNTIME['white_spectra'] = 1
                _RUNTIME['selecting_white'] = False
                print(f"White spectrum saved.")
                # del answer
            else:
                print(f"No selecting anything")

        else:
            print(f"We have a click at ({x},{y})")

            # Else it was just a click and we can reset x0 and y0
            _RUNTIME['rect_x_0'] = None
            _RUNTIME['rect_y_0'] = None
            pixel = _RUNTIME['img_array'][int(y), int(x)]
            update_px_plot(spectrum=pixel, x0=x, y0=y)

        print(f"Mouse button {eventi.button} released at ({x},{y})")


def update_px_plot(spectrum: np.array, std: np.array=None, x0=None, y0=None, x1=None, y1=None):
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

    if spectrum is None:
        print(f"WARNING: Cannot update pixel plot: spectrum was None. Returning.")
        return

    # Draw new plot and refersh canvas
    _RUNTIME['fig_agg_px_plot'].get_tk_widget().forget()
    ax_px_plot.plot(spectrum)

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
    else:
        print(f"WARNING: Bad pixel plot update call.")


def update_false_color_canvas():

    rgb_bands = (_RUNTIME['band_R'], _RUNTIME['band_G'], _RUNTIME['band_B'])

    if rgb_bands[0]==0 and rgb_bands[1]==0 and rgb_bands[2]==0:
        print(f"Trying to get RGB bands from HDR file.")
        if _RUNTIME['cube_data'] is not None:
            rgb_bands = [int(band) - 1 for band in _RUNTIME['cube_data'].metadata['default bands']]

    _RUNTIME['band_R'] = rgb_bands[0]
    _RUNTIME['band_G'] = rgb_bands[1]
    _RUNTIME['band_B'] = rgb_bands[2]

    view_mode = _RUNTIME['view_mode']

    def autoscale_int_image(image: np.array) -> np.array:

        cmax = np.max(image)
        print(f"Cube max value: {cmax} in autoscale.")

        if cmax > 255:
            print(f"Autoscaling false color image.")
            autoscaled = np.divide(image, cmax)
            autoscaled = autoscaled * 255
            autoscaled = autoscaled.astype(np.uint16)
            return autoscaled
        else:
            return image.astype(np.uint16)

    def autoscale_float_image(image: np.array) -> np.array:
        """Not sure if this is even needed because we should have white reference."""

        cmax = np.max(image)
        print(f"Cube max value: {cmax} in autoscale.")

        if cmax > 1.0:
            print(f"Autoscaling false color image.")
            autoscaled = np.divide(image, cmax, dtype=np.float32)
            return autoscaled
        else:
            return image.astype(np.float32)

    if view_mode == 'cube' and not _RUNTIME['selecting_dark'] and not _RUNTIME['selecting_white']:
        if _RUNTIME['img_array'] is None:
            print(f"Image array None. Nothing to show.")
            return

        if _RUNTIME['white_corrected']:
            false_color_rgb = _RUNTIME['img_array'][:, :, rgb_bands].astype(np.float32)
        else:
            false_color_rgb = _RUNTIME['img_array'][:, :, rgb_bands].astype(np.uint16)
            false_color_rgb = autoscale_int_image(false_color_rgb)

    elif view_mode == 'dark' or _RUNTIME['selecting_dark']:

        if _RUNTIME['img_array_dark'] is None:
            print(f"Image array None. Nothing to show.")
            return

        false_color_rgb = _RUNTIME['img_array_dark'][:, :, rgb_bands].astype(np.uint16)
        false_color_rgb = autoscale_int_image(false_color_rgb)

    elif view_mode == 'white' or _RUNTIME['selecting_white']:

        if _RUNTIME['img_array_white'] is None:
            print(f"Image array None. Nothing to show.")
            return

        false_color_rgb = _RUNTIME['img_array_white'][:, :, rgb_bands].astype(np.uint16)
        false_color_rgb = autoscale_int_image(false_color_rgb)
    else:
        print(f"WARNING: unknown view mode '{view_mode}' and/or selection combination selecting "
              f"dark={_RUNTIME['selecting_dark']}, white={_RUNTIME['selecting_white']}.")
        return

    _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
    ax_false_color.imshow(false_color_rgb)
    ax_false_color.set_xlabel('samples')
    ax_false_color.set_ylabel('lines')

    _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def cube_meta():

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
            # print(f"len wls: {len(_VARS['cube_wls'])}")
            # print(f"len bands: {len(_VARS['cube_bands'])}")
            walength_set = True
        elif key.lower() == 'fwhm':
            continue
        else:
            window[guiek_cube_meta_text].print(f"{key}: {value}")

    if not walength_set:
        print(f"Could not set wavelengths and bands.")
        return

    ax_px_plot.set_xlabel('Band')
    bands = _RUNTIME['cube_bands']
    wls = _RUNTIME['cube_wls']

    ax_px_plot.set_xlim(bands[0], bands[-1])

    def forward(x):
        return np.interp(x, bands, wls)

    def inverse(x):
        return np.interp(x, wls, bands)

    # FIXME the wavelength axis is not perfect as two figures are shown at the extremes
    secax = ax_px_plot.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel(r"Wavelength [$nm$]")


def find_cube(user_selected_file_path):
    """

    FIXME make independent of the _RUNTIME selecting_white etc.

    :param user_selected_file_path:
    :return:
    """
    selected_dir_path = os.path.dirname(user_selected_file_path)
    selected_file_name = os.path.basename(user_selected_file_path)
    base_name = selected_file_name.rsplit(sep='.',maxsplit=1)[0]

    print(f"File: '{selected_file_name}' in '{selected_dir_path}'. Base name is '{base_name}'.")

    hdr_found = False
    raw_found = False
    hdr_file_name = None
    raw_file_name = None

    file_list = os.listdir(selected_dir_path)
    for file_name in file_list:
        if file_name.startswith(base_name) and file_name.lower().endswith(".hdr"):
            hdr_found = True
            hdr_file_name = file_name
        if file_name.startswith(base_name) and (file_name.lower().endswith(".raw") or file_name.lower().endswith(".img")):
            raw_found = True
            raw_file_name = file_name

    if hdr_found and raw_found:
        print(f"Envi cube files OK. ")
        hdr_path = os.path.join(selected_dir_path, hdr_file_name)
        raw_path = os.path.join(selected_dir_path, raw_file_name)

        if _RUNTIME['selecting_dark']:
            _STATE['dark_cube_hdr_path'] = hdr_path
            _STATE['dark_cube_data_path'] = raw_path
        if _RUNTIME['selecting_white']:
            _STATE['white_cube_hdr_path'] = hdr_path
            _STATE['white_cube_data_path'] = raw_path
        else:
            _STATE['main_cube_hdr_path'] = hdr_path
            _STATE['main_cube_data_path'] = raw_path

        open_cube(hdr_path=hdr_path, data_path=raw_path)
    else:
        print(f"Not OK. Either hdr or raw file not found from given directory.")


def open_cube(hdr_path, data_path):

    if hdr_path is None or data_path is None:
        print(f"Either HDR path or DATA path was None. Cannot open cube.")
        return

    cube_data = envi.open(file=hdr_path, image=data_path)
    img_array = cube_data.load().asarray()

    if not _RUNTIME['selecting_dark'] and not _RUNTIME['selecting_white']:

        _RUNTIME['cube_data'] = cube_data
        _RUNTIME['img_array'] = img_array

        # First set things up using metadata
        cube_meta()

    elif _RUNTIME['selecting_dark']:
        _RUNTIME['img_array_dark'] = img_array
    elif _RUNTIME['selecting_white']:
        _RUNTIME['img_array_white'] = img_array
    else:
        print(f"WARNING: WE SHOULD NOT BE HERE!!")

    # Connect mouse click
    if not _RUNTIME['mouse_handlers_connected']:
        cid_press = fig_false_color.canvas.mpl_connect('button_press_event', mouse_click_event)
        cid_release = fig_false_color.canvas.mpl_connect('button_release_event', mouse_release_event)
        _RUNTIME['mouse_handlers_connected'] = True

    # Draw cube to canvas
    update_false_color_canvas()


def calc_dark():

    print(f"Dark calculation called...")

    if _RUNTIME['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    dark_spectrum = _RUNTIME['dark_median']
    if dark_spectrum is None:
        print(f"Cannot calculate dark because dark spectrum is None. Select a region from a dark cube first.")
        return

    # FIXME subtract the MEDIAN of the dark. Not the mean
    _RUNTIME['img_array'] = _RUNTIME['img_array'] - dark_spectrum
    _RUNTIME['img_array'] = np.clip(_RUNTIME['img_array'], a_min=0, a_max=None)
    _RUNTIME['img_array'] = _RUNTIME['img_array']

    update_false_color_canvas()


def calc_white():

    print(f"White calculation called...")

    # img_array = _VARS['img_array']
    if _RUNTIME['img_array'] is None:
        print(f"Cannot calculate dark because image array is None. Select a cube first.")
        return

    white_spectrum = _RUNTIME['white_spectra']
    if white_spectrum is None:
        print(f"Cannot calculate dark because dark spectrum is None. Select a region from a dark cube first.")
        return

    _RUNTIME['img_array'] = np.divide(_RUNTIME['img_array'], white_spectrum, dtype=np.float32)
    _RUNTIME['white_corrected'] = True
    # _VARS['img_array'] = img_array

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

    _RUNTIME['rectangle_handles'] = []
    _RUNTIME['dot_handles'] = []
    update_false_color_canvas()


state_dir_path = os.getcwd()
state_file_name = "ci_state"


def state_save():

    _STATE['band_R'] = _RUNTIME['band_R']
    _STATE['band_G'] = _RUNTIME['band_G']
    _STATE['band_B'] = _RUNTIME['band_B']

    TH.write_dict_as_toml(dictionary=_STATE, directory=state_dir_path, filename=state_file_name)
    print(f"State saved")


def state_load():
    state = TH.read_toml_as_dict(directory=state_dir_path, filename=state_file_name)
    for key, value in state.items():
        _STATE[key] = value

    _RUNTIME['band_R'] = _STATE['band_R']
    _RUNTIME['band_G'] = _STATE['band_G']
    _RUNTIME['band_B'] = _STATE['band_B']

    #We have _RUNTIME at this point so we can update the RGB inboxes as well
    window[guiek_r_input].update(str(_RUNTIME['band_R']))
    window[guiek_g_input].update(str(_RUNTIME['band_G']))
    window[guiek_b_input].update(str(_RUNTIME['band_B']))


# GUI entity keys
guiek_cube_file_selected = "-CUBE SELECT-"
guiek_dark_file_selected = "-DARK SELECT-"
guiek_white_file_selected = "-WHITE SELECT-"
guiek_cube_show_filename = "-CUBE SHOW FILENAME-"
guiek_dark_show_filename = "-DARK SHOW FILENAME-"
guiek_white_show_filename = "-WHITE SHOW FILENAME-"

# For show buttons
guiek_cube_show_button = "-CUBE SHOW BUTTON-"
guiek_dark_show_button = "-DARK SHOW BUTTON-"
guiek_white_show_button = "-WHITE SHOW BUTTON-"

# RGB selection
guiek_r_input = "-R-"
guiek_g_input = "-G-"
guiek_b_input = "-B-"
guiek_rgb_update_button = "-RGB UPDATE BUTTON-"

guiek_calc_dark = "-CALCULATE DARK-"
guiek_calc_white = "-CALCULATE WHITE-"
guiek_cube_false_color = "-FALSE COLOR-"
guiek_pixel_plot_canvas = "-PX PLOT CANVAS-"
guiek_cube_meta_text = "-CUBE META-"

fig_px_plot = None

fig_px_plot = plt.figure(figsize=(5, 4), dpi=100)
ax_px_plot = fig_px_plot.add_subplot(111)

fig_false_color = plt.figure(figsize=(5, 4), dpi=100)
ax_false_color = fig_false_color.add_subplot(111)


cube_meta_column = [
    [
        sg.Text("Cube metadata"),
    ],
    [
        sg.Multiline(size=(70, 25), key=guiek_cube_meta_text),
    ]
]

cube_column = [
    [
        sg.Text("Cube"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_cube_show_filename, disabled=True, text_color='black'),
        sg.FileBrowse(key=guiek_cube_file_selected, target=guiek_cube_file_selected, enable_events=True,),
        sg.Button('Show', enable_events=True, key=guiek_cube_show_button),
    ],
    [
        sg.Text("Dark"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_dark_show_filename, disabled=True, text_color='black'),
        sg.FileBrowse(key=guiek_dark_file_selected, target=guiek_dark_file_selected, enable_events=True,),
        sg.Button('Show', enable_events=True, key=guiek_dark_show_button),
        sg.Button("Calculate", k=guiek_calc_dark),
    ],
    [
        sg.Text("White"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_white_show_filename, disabled=True, text_color='black', background_color='grey'),
        sg.FileBrowse(key=guiek_white_file_selected, target=guiek_white_file_selected, enable_events=True,),
        sg.Button('Show', enable_events=True, key=guiek_white_show_button),
        sg.Button("Calculate", k=guiek_calc_white),
    ],
    [
        # sg.Listbox(values=[], enable_events=True, size=(80, 20), key=guiek_file_list)
        # sg.Text("Cube false color"),
        sg.Canvas(key=guiek_cube_false_color),
    ],
    [
        sg.Text("R"),
        sg.In(size=(5, 1), key=guiek_r_input),
        sg.Text("G"),
        sg.In(size=(5, 1), key=guiek_g_input),
        sg.Text("B"),
        sg.In(size=(5, 1), key=guiek_b_input),
        sg.Button('Update', enable_events=True, key=guiek_rgb_update_button)
    ],
]

pixel_plot_column = [
    # [sg.Text("Choose an ENVI cube directory from list on left:")],
    [sg.Canvas(key=guiek_pixel_plot_canvas)],
    [sg.Button("Clear")]
]

layout = [
    [
        sg.Column(cube_meta_column),
        sg.VSeperator(),
        sg.Column(cube_column),
        sg.VSeperator(),
        sg.Column(pixel_plot_column),
    ]
]

window = sg.Window("Cube Inspector", layout=layout, margins=(100,100), finalize=True)


# TODO Incorporate dark and white selection logic with buttons.
# TODO Show corresponding wl for RGB band selection. In plot or as a text box?
# TODO Gaussian or mean RGB.

# Keep most of the global stuff in this single dictionary for later access
_RUNTIME = {
    'window': window,
    'fig_agg_px_plot': draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot),
    'fig_agg_false_color': draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color),
    'pltFig': False,
    'cube_data': None,
    'img_array': None,
    'img_array_dark': None,
    'img_array_white': None,
    'white_corrected': False,
    'cube_wls': None,
    'cube_bands': None,
    'rect_x_0': None,
    'rect_y_0': None,
    'rect_x_1': None,
    'rect_y_1': None,

    'rectangle_handles': [],
    'dot_handles': [],

    'selecting_white': False,
    'selecting_dark': False,

    'view_mode': None, # 'cube', 'dark', 'white' or None

    'white_spectra': None,
    'dark_median': None,

    'mouse_handlers_connected': False,

    'band_B': 0,
    'band_G': 0,
    'band_R': 0,
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


def get_base_name_wo_postfix(path: str) -> str:
    """Returns the file name in given path without the postfix such as .hdr."""

    base_name = os.path.basename(path).rsplit('.', maxsplit=1)[0]
    return base_name


def restore_from_previous_session():

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

    # _RUNTIME['selecting_dark'] = True
    # open_cube(hdr_path=_STATE['dark_cube_hdr_path'], data_path=_STATE['dark_cube_data_path'])
    # _RUNTIME['selecting_dark'] = False
    # _RUNTIME['selecting_white'] = True
    # open_cube(hdr_path=_STATE['white_cube_hdr_path'], data_path=_STATE['white_cube_data_path'])
    # _RUNTIME['selecting_white'] = False
    # open_cube(hdr_path=_STATE['main_cube_hdr_path'], data_path=_STATE['main_cube_data_path'])

    _RUNTIME['view_mode'] = 'cube'
    update_false_color_canvas()


def handle_cube_file_selected(file_path:str):
    _RUNTIME['selecting_white'] = False
    _RUNTIME['selecting_dark'] = False
    _RUNTIME['view_mode'] = 'cube'
    find_cube(file_path)
    update_false_color_canvas()


def handle_dark_file_selected(file_path: str):
    _RUNTIME['selecting_dark'] = True
    _RUNTIME['view_mode'] = 'dark'

    find_cube(file_path)
    # image_array_dark should now have a value
    dark_cube = _RUNTIME['img_array_dark']
    print(f"Dark cube set. Calculating median of the scan lines.")

    # Scan lines are on axis=0
    med = np.median(dark_cube, axis=0)
    print(f"DEBUG: dark median shape: {med.shape}")
    _RUNTIME['dark_median'] = med

    _RUNTIME['selecting_dark'] = False
    print(f"Dark median saved.")
    update_false_color_canvas()


def handle_white_file_selected(file_path: str):
    _RUNTIME['selecting_white'] = True
    _RUNTIME['view_mode'] = 'white'
    find_cube(file_path)
    _RUNTIME['selecting_white'] = False
    update_false_color_canvas()


def main():
    answer = sg.popup_yes_no("Wanna load previous session?")
    if answer == 'Yes':
        restore_from_previous_session()

    # Infinite GUI loop
    while True:
        event, values = window.read()
        print(f"event {event}, values: {values}")

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "Clear":
            clear_plot()

        if event == guiek_cube_file_selected:
            window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(values[guiek_cube_file_selected]))
            handle_cube_file_selected(values[guiek_cube_file_selected])

        if event == guiek_dark_file_selected:
            window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(values[guiek_dark_file_selected]))
            handle_dark_file_selected(values[guiek_dark_file_selected])

        if event == guiek_white_file_selected:
            window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(values[guiek_white_file_selected]))
            handle_white_file_selected(values[guiek_white_file_selected])

        if event == guiek_cube_show_button:
            _RUNTIME['view_mode'] = 'cube'
            print(f"Cube show button press")
            update_false_color_canvas()

        if event == guiek_dark_show_button:
            _RUNTIME['view_mode'] = 'dark'
            print(f"Dark show button press")
            update_false_color_canvas()

        if event == guiek_white_show_button:
            _RUNTIME['view_mode'] = 'white'
            print(f"White show button press")
            update_false_color_canvas()

        if event == guiek_calc_dark:
            print(f"Dark button press")
            calc_dark()

        if event == guiek_calc_white:
            print(f"White button press")
            calc_white()

        if event == guiek_rgb_update_button:
            print(f"RGB update button pressed")
            try:
                _RUNTIME['band_R'] = int(values[guiek_r_input])
                _RUNTIME['band_G'] = int(values[guiek_g_input])
                _RUNTIME['band_B'] = int(values[guiek_b_input])
                update_false_color_canvas()
            except ValueError as ve:
                print(f"Failed to cast to int.")

    state_save()
    window.close()


if __name__ == '__main__':

    try:
        state_load()
        print(f"State loaded: {_STATE}")
    except FileNotFoundError as e:
        print(f"No previous state found.")

    main()

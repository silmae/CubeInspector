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

    if eventi.button == 1:
        x0 = _RUNTIME['rect_x_0']
        y0 = _RUNTIME['rect_y_0']

        y = eventi.ydata
        x = eventi.xdata

        drag_treshold = 5 # pixels FIXME put to settings

        # We have a drag if we have previously set (and not cleared) x0 and y0 and the release position is far enough away.
        if (x0 is not None or y0 is not None) and (math.fabs(x0 - x) > drag_treshold or math.fabs(y0 - y) > drag_treshold):

            print(f"Mouse dragged from ({x0}, {y0}) to ({x}, {y})")
            drag_start_x = int(min(x, x0))
            drag_start_y = int(min(y, y0))
            drag_end_x = int(max(x, x0))
            drag_end_y = int(max(y, y0))
            rows = list(np.arange(start=drag_start_x, stop=drag_end_x, step=1))
            cols = list(np.arange(start=drag_start_y, stop=drag_end_y, step=1))
            print(f"Rows: {rows}")
            print(f"Cols: {cols}")

            # So for some reason, when reading from image array, rows and colums are inverted
            # (compared to when reading from 'cube_data').
            # sub_image = _VARS['img_array'].read_subimage(rows=cols, cols=rows)
            sub_image = _RUNTIME['img_array'][cols][:, rows]

            # sg.popup_ok(title="Wait a bit, I'm calculating..")
            sub_mean = np.mean(sub_image, axis=(0,1))
            sub_std = np.std(sub_image, axis=(0,1))

            print(f"len mean spectra: {len(sub_mean)}")

            # Draw new plot and refersh canvas
            _RUNTIME['fig_agg'].get_tk_widget().forget()
            ax_px_plot.plot(sub_mean)
            ax_px_plot.fill_between(_RUNTIME['cube_bands'], sub_mean - (sub_std / 2), sub_mean + (sub_std / 2), alpha=0.2)
            _RUNTIME['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

            # Draw rectangle over RGB canvas and refresh
            width = math.fabs(drag_start_x - drag_end_x)
            height = math.fabs(drag_start_y - drag_end_y)
            _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
            handle = ax_false_color.add_patch(Rectangle((drag_start_x, drag_start_y), width=width, height=height, fill=False, edgecolor='gray'))
            _RUNTIME['rectangle_handles'].append(handle)
            _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

            if _RUNTIME['selecting_dark']:

                # answer = sg.popup_yes_no("Do you want to save selected mean spectra as dark reference?")
                # if answer == 'Yes':
                # FIXME Apparently, we cannot raise a popup iside a Matplotlib event without freezing the UI.

                _RUNTIME['dark_spectra'] = np.median(sub_image, axis=(0, 1))
                _RUNTIME['dark_spectra'] = 1
                _RUNTIME['selecting_dark'] = False
                print(f"Dark spectrum saved.")
                # del answer
            elif _RUNTIME['selecting_white']:
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

            # And do the normal click stuff
            _RUNTIME['fig_agg'].get_tk_widget().forget()
            # pixel = _VARS['img_array'].read_pixel(row=int(y), col=int(x))
            pixel = _RUNTIME['img_array'][int(y), int(x)]
            ax_px_plot.plot(pixel)
            _RUNTIME['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

            # Draw click location as a dot
            _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
            handle = ax_false_color.scatter(int(x), int(y))
            _RUNTIME['dot_handles'].append(handle)
            _RUNTIME['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

    print(f"Mouse button {eventi.button} released at ({x},{y})")


def update_false_color_canvas():
    cube_data = _RUNTIME['cube_data']

    if cube_data is None:
        return

    _RUNTIME['fig_agg_false_color'].get_tk_widget().forget()
    default_bands = [int(band) - 1 for band in cube_data.metadata['default bands']]

    if _RUNTIME['selecting_dark']:
        false_color_rgb = _RUNTIME['img_array_dark'][:, :, default_bands].astype(np.uint16)
    elif _RUNTIME['selecting_white']:
        false_color_rgb = _RUNTIME['img_array_white'][:, :, default_bands].astype(np.uint16)
    else:
        # false_color_rgb = cube_data.read_bands(bands=default_bands)
        # false_color_rgb = _VARS['img_array'].read_bands(bands=default_bands).astype(np.uint16)
        if _RUNTIME['white_corrected']:
            false_color_rgb = _RUNTIME['img_array'][:, :, default_bands].astype(np.float32)
        else:
            false_color_rgb = _RUNTIME['img_array'][:, :, default_bands].astype(np.uint16)

    ######## False color canvas
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

    dark_spectrum = _RUNTIME['dark_spectra']
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
    _RUNTIME['fig_agg'].get_tk_widget().forget()
    ax_px_plot.clear()
    _RUNTIME['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

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
    TH.write_dict_as_toml(dictionary=_STATE, directory=state_dir_path, filename=state_file_name)
    print(f"State saved")


def state_load():
    state = TH.read_toml_as_dict(directory=state_dir_path, filename=state_file_name)
    for key, value in state.items():
        _STATE[key] = value


# GUI entity keys
guiek_cube_file_selected = "-CUBE SELECT-"
guiek_dark_file_selected = "-DARK SELECT-"
guiek_white_file_selected = "-WHITE SELECT-"
guiek_cube_show_filename = "-CUBE SHOW FILENAME-"
guiek_dark_show_filename = "-DARK SHOW FILENAME-"
guiek_white_show_filename = "-WHITE SHOW FILENAME-"
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
    ],
    [
        sg.Text("Dark"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_dark_show_filename, disabled=True, text_color='black'),
        sg.FileBrowse(key=guiek_dark_file_selected, target=guiek_dark_file_selected, enable_events=True,),
        sg.Button("Calculate", k=guiek_calc_dark),
    ],
    [
        sg.Text("White"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_white_show_filename, disabled=True, text_color='black', background_color='grey'),
        sg.FileBrowse(key=guiek_white_file_selected, target=guiek_white_file_selected, enable_events=True,),
        sg.Button("Calculate", k=guiek_calc_white),
    ],
    [
        # sg.Listbox(values=[], enable_events=True, size=(80, 20), key=guiek_file_list)
        # sg.Text("Cube false color"),
        sg.Canvas(key=guiek_cube_false_color),
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

# try:
#     _STATE = state_load()
# except FileNotFoundError as e:
#
#     print(f"Could not find state file. Initializing with defaults.")


# Keep most of the global stuff in this single dictionary for later access
_RUNTIME = {
    'window': window,
    'fig_agg': draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot),
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

    'white_spectra': None,
    'dark_spectra': None,

    'mouse_handlers_connected': False,
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


def get_base_name_wo_postfix(path: str) -> str:
    """Returns the file name in given path without the postfix such as .hdr."""

    base_name = os.path.basename(path).rsplit('.', maxsplit=1)[0]
    return base_name


def restore_from_previous_session():

    print(f"Trying to restore state from previous session.")

    if _STATE['main_cube_hdr_path'] is not None:
        window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(_STATE['main_cube_hdr_path']))
    if _STATE['dark_cube_hdr_path'] is not None:
        window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(_STATE['dark_cube_hdr_path']))
    if _STATE['white_cube_hdr_path'] is not None:
        window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(_STATE['white_cube_hdr_path']))

    open_cube(hdr_path=_STATE['main_cube_hdr_path'], data_path=_STATE['main_cube_data_path'])
    open_cube(hdr_path=_STATE['dark_cube_hdr_path'], data_path=_STATE['dark_cube_data_path'])
    open_cube(hdr_path=_STATE['white_cube_hdr_path'], data_path=_STATE['white_cube_data_path'])

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
            _RUNTIME['selecting_white'] = False
            _RUNTIME['selecting_dark'] = False
            find_cube(values[guiek_cube_file_selected])

        if event == guiek_dark_file_selected:
            window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(values[guiek_dark_file_selected]))
            _RUNTIME['selecting_dark'] = True
            find_cube(values[guiek_dark_file_selected])

        if event == guiek_white_file_selected:
            window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(values[guiek_white_file_selected]))
            _RUNTIME['selecting_white'] = True
            find_cube(values[guiek_white_file_selected])

        if event == guiek_calc_dark:
            print(f"Dark button press")
            calc_dark()

        if event == guiek_calc_white:
            print(f"White button press")
            calc_white()

    state_save()
    window.close()


if __name__ == '__main__':

    try:
        state_load()
        print(f"State loaded: {_STATE}")
    except FileNotFoundError as e:
        print(f"No previous state found.")

    main()

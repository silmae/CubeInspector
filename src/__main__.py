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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import AutoMinorLocator

matplotlib.use("TkAgg")

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
    _VARS['rect_x_0'] = eventi.xdata
    _VARS['rect_y_0'] = eventi.ydata


def mouse_release_event(eventi):

    if eventi.button == 1:
        x0 = _VARS['rect_x_0']
        y0 = _VARS['rect_y_0']

        y = eventi.ydata
        x = eventi.xdata

        drag_treshold = 2 # pixels

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
            sub_image = _VARS['cube_data'].read_subimage(rows=rows, cols=cols)

            # sg.popup_ok(title="Wait a bit, I'm calculating..")
            sub_mean = np.mean(sub_image, axis=(0,1))

            print(f"len mean spectra: {len(sub_mean)}")

            # Draw new plot and refersh canvas
            _VARS['fig_agg'].get_tk_widget().forget()
            ax_px_plot.plot(sub_mean)
            _VARS['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)


        else:

            print(f"We have a click at ({x},{y})")

            # Else it was just a click and we can reset x0 and y0
            _VARS['rect_x_0'] = None
            _VARS['rect_y_0'] = None

            # And do the normal click stuff
            _VARS['fig_agg'].get_tk_widget().forget()
            pixel = _VARS['cube_data'].read_pixel(row=int(y), col=int(x))
            ax_px_plot.plot(pixel)
            _VARS['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

    print(f"Mouse button {eventi.button} released at ({x},{y})")


def update_false_color_canvas():
    _VARS['fig_agg_false_color'].get_tk_widget().forget()
    cube_data = _VARS['cube_data']
    default_bands = [int(band) - 1 for band in cube_data.metadata['default bands']]
    false_color_rgb = cube_data.read_bands(bands=default_bands)

    ######## False color canvas
    ax_false_color.imshow(false_color_rgb)
    ax_false_color.set_xlabel('samples')
    ax_false_color.set_ylabel('lines')

    _VARS['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def cube_meta():

    cube_data = _VARS['cube_data']
    for key,value in cube_data.metadata.items():
        if key == 'wavelength':
            _VARS['cube_wls'] = np.array(list(float(v) for v in value))
            # print(_VARS['cube_wls'])
            _VARS['cube_bands'] = np.arange(start=0, stop=len(value), step=1)
            # print(f"len wls: {len(_VARS['cube_wls'])}")
            # print(f"len bands: {len(_VARS['cube_bands'])}")
        elif key == 'fwhm':
            continue
        else:
            window[guiek_cube_meta_text].print(f"{key}: {value}")

    ax_px_plot.set_xlabel('Band')
    bands = _VARS['cube_bands']
    wls = _VARS['cube_wls']

    ax_px_plot.set_xlim(bands[0], bands[-1])

    def forward(x):
        return np.interp(x, bands, wls)

    def inverse(x):
        return np.interp(x, wls, bands)

    # FIXME the wavelength axis is not perfect as two figures are shown at the extremes
    secax = ax_px_plot.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel(r"Wavelength [$nm$]")


def open_cube(user_selected_file_path):
    selected_dir_path = os.path.dirname(user_selected_file_path)
    selected_file_name = os.path.basename(user_selected_file_path)

    print(f"File: '{selected_file_name}' in '{selected_dir_path}'.")

    hdr_found = False
    raw_found = False
    hdr_file_name = None
    raw_file_name = None

    file_list = os.listdir(selected_dir_path)
    for file_name in file_list:
        if file_name.endswith(".hdr"):
            hdr_found = True
            hdr_file_name = file_name
        # TODO check for .img (alias for raw, apparently)
        if file_name.endswith(".raw"):
            raw_found = True
            raw_file_name = file_name

    if hdr_found and raw_found:
        print(f"Envi cube files OK. ")
        hdr_path = os.path.join(selected_dir_path, hdr_file_name)
        raw_path = os.path.join(selected_dir_path, raw_file_name)
        cube_data = envi.open(file=hdr_path, image=raw_path)
        _VARS['cube_data'] = cube_data

        # First set things up using metadata
        cube_meta()

        # Draw cube to canvas
        update_false_color_canvas()
        # Connect mouse click
        cid_press = fig_false_color.canvas.mpl_connect('button_press_event', mouse_click_event)
        cid_release = fig_false_color.canvas.mpl_connect('button_release_event', mouse_release_event)


    else:
        print(f"Not OK. Either hdr or raw file not found from given directory.")


def clear_plot():
    """Clear pixel plot.

    TODO maybe make more general to clear any pyplot axis?
    """

    print(f"Clearing pixel plot")
    _VARS['fig_agg'].get_tk_widget().forget()
    ax_px_plot.clear()
    _VARS['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)


# GUI entity keys
guiek_cube_file_selected = "-DIR-"
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
        sg.Text("This ought to contain metadata"),
    ],
    [
        sg.Multiline(size=(100, 25), key=guiek_cube_meta_text),
    ]
]

cube_column = [
    [
        sg.Text("Select ENVI cube directory"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_cube_file_selected),
        # sg.FolderBrowse(),
        sg.FileBrowse()
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

window = sg.Window("Cube Inspector", layout=layout, margins=(500,300), finalize=True)

# Keep most of the global stuff in this single dictionary for later access
_VARS = {'window': window,
         'fig_agg': draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot),
         'fig_agg_false_color': draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color),
         'pltFig': False,
         'cube_data': None,
         'cube_wls': None,
         'cube_bands': None,
         'rect_x_0': None,
         'rect_y_0': None,
         'rect_x_1': None,
         'rect_y_1': None,
         }


# Infinite GUI loop
while True:
    event, values = window.read()
    print(f"event {event}, values: {values}")

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Clear":
        clear_plot()

    if event == guiek_cube_file_selected:
        open_cube(values[guiek_cube_file_selected])


window.close()

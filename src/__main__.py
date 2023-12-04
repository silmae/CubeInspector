"""

Main GUI loop and other stuff needed for that.

Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/
Github project PySimpleGUI: https://github.com/PySimpleGUI/PySimpleGUI

Refreshing plot: https://gist.github.com/KenoLeon/e913de9e1fe690ebe287e6d1e54e3b97
"""

import os

import PySimpleGUI as sg
import spectral.io.envi as envi
import spectral as spy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use("TkAgg")


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
    y = eventi.xdata
    x = eventi.ydata

    # Can be none if clicked outside of the image area.
    # TODO check which canvas was cliced?
    if x is not None and y is not None:
        _VARS['fig_agg'].get_tk_widget().forget()
        pixel = _VARS['cuba_data'].read_pixel(row=int(x), col=int(y))
        ax_px_plot.plot(pixel)
        _VARS['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)
    else:
        print(f"Mouse click out of image area.")


def update_false_color_canvas():
    _VARS['fig_agg_false_color'].get_tk_widget().forget()
    cube_data = _VARS['cuba_data']
    default_bands = [int(band) - 1 for band in cube_data.metadata['default bands']]
    false_color_rgb = cube_data.read_bands(bands=default_bands)

    ######## False color canvas
    ax_false_color.imshow(false_color_rgb)
    _VARS['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

    for key,value in cube_data.metadata.items():
        if key == 'wavelength' or key == 'fwhm':
            continue
        else:
            window[guiek_cube_meta_text].print(f"{key}: {value}")


def clear_plot():
    """Clear pixel plot.

    TODO maybe make more general to clear any pyplot axis?
    """

    print(f"Clearing pixel plot")
    _VARS['fig_agg'].get_tk_widget().forget()
    ax_px_plot.clear()
    _VARS['fig_agg'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)


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
        _VARS['cuba_data'] = cube_data
        # Draw cube to canvas
        update_false_color_canvas()
        # Connect mouse click
        cid = fig_false_color.canvas.mpl_connect('button_press_event', mouse_click_event)

    else:
        print(f"Not OK. Either hdr or raw file not found from given directory.")


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
         }


# Infinite GUI loop
while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Clear":
        clear_plot()

    if event == guiek_cube_file_selected:
        open_cube(values[guiek_cube_file_selected])


window.close()

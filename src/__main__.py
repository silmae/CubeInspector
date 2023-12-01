
import os

import PySimpleGUI as sg
import spectral.io.envi as envi
import spectral as spy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use("TkAgg")


# GUI entity keys
guiek_cube_dir_selected = "-DIR-"
# guiek_file_list = "-FILE LIST-"
guiek_cube_false_color = "-FALSE COLOR-"


"""
Refreshing plot: https://gist.github.com/KenoLeon/e913de9e1fe690ebe287e6d1e54e3b97
"""

click_x = 0
click_y = 0

fig_px_plot = None
cube_data = None
fig_px_plot = plt.figure(figsize=(5, 4), dpi=100)
ax_px_plot = fig_px_plot.add_subplot(111)


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_cube_dir_selected),
        sg.FolderBrowse(),
    ],
    [
        # sg.Listbox(values=[], enable_events=True, size=(80, 20), key=guiek_file_list)
        sg.Text("Cube false color"),
        sg.Canvas(key=guiek_cube_false_color),
    ],
]

image_viewer_column = [
    [sg.Text("Choose an ENVI cube directory from list on left:")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Button("Clear")]
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


window = sg.Window("Cube Inspector", layout=layout, margins=(500,300), finalize=True)

_VARS = {'window': False,
         'fig_agg': draw_figure(window["-CANVAS-"].TKCanvas, fig_px_plot),
         'pltFig': False}

# _VARS['fig_agg'] = draw_figure(window["-CANVAS-"].TKCanvas, fig_px_plot)


def mouse_event(eventi):
    print('x: {} and y: {}'.format(eventi.xdata, eventi.ydata))

    y = eventi.xdata
    x = eventi.ydata

    if x is not None and y is not None:
        _VARS['fig_agg'].get_tk_widget().forget()
        pixel = cube_data.read_pixel(row=int(x), col=int(y))
        ax_px_plot.plot(pixel)
        _VARS['fig_agg'] = draw_figure(window["-CANVAS-"].TKCanvas, fig_px_plot)
    else:
        print(f"Mouse click out of image area.")


def clear_plot():
    print(f"Clearing pixel plot")
    _VARS['fig_agg'].get_tk_widget().forget()
    ax_px_plot.clear()
    _VARS['fig_agg'] = draw_figure(window["-CANVAS-"].TKCanvas, fig_px_plot)


while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Clear":
        clear_plot()

    if event == guiek_cube_dir_selected:

        cube_dir = values[guiek_cube_dir_selected]

        hdr_found = False
        raw_found = False
        hdr_file_name = None
        raw_file_name = None

        cube_data = None

        file_list = os.listdir(cube_dir)
        for file_name in file_list:
            if file_name.endswith(".hdr"):
                hdr_found = True
                hdr_file_name = file_name
            if file_name.endswith(".raw"):
                raw_found = True
                raw_file_name = file_name

        if hdr_found and raw_found:
            print(f"Envi cube files OK. ")
            hdr_path = os.path.join(cube_dir,hdr_file_name)
            raw_path = os.path.join(cube_dir,raw_file_name)
            cube_data = envi.open(file=hdr_path,image=raw_path)
            default_bands = [int(band) - 1 for band in cube_data.metadata['default bands']]
            false_color_rgb = cube_data.read_bands(bands=default_bands)

            ######## False color canvas
            fig_false_color = plt.figure(figsize=(5, 4), dpi=100)
            ax_false_color = fig_false_color.add_subplot(111)
            ax_false_color.imshow(false_color_rgb)
            # Connect mouse click
            cid = fig_false_color.canvas.mpl_connect('button_press_event', mouse_event)

            draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)
        else:
            print(f"Not OK. Either hdr or raw file not found from given directory.")




window.close()

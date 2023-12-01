
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
guiek_cube_dir = "-DIR-"
guiek_file_list = "-FILE LIST-"


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key=guiek_cube_dir),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(80, 20), key=guiek_file_list)
    ],
]

image_viewer_column = [
    [sg.Text("Choose an ENVI cube directory from list on left:")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Button("OK")]
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Cube Inspector", layout=layout, margins=(500,300), finalize=True)


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


fig = None
cube_data = None

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Folder name was filled in, make a list of files in the folder
    if event == guiek_cube_dir:

        cube_dir = values[guiek_cube_dir]
        # print(f"Directory: {cube_dir}")

        try:
            # Get list of files in folder
            file_list = os.listdir(cube_dir)
            # print(f"file_list: {file_list}")

            hdr_found = False
            raw_found = False
            hdr_file_name = None
            raw_file_name = None

            cube_data = None

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

                # Add the plot to the window
                # draw_figure(window["-CANVAS-"].TKCanvas, fig)

                print(f"Cube meta: {cube_data.metadata}")
            else:
                print(f"Not OK. Either hdr or raw file not found from given directory.")
        except:
            file_list = []

        fnames = [
            f for f in file_list
            if os.path.isfile(os.path.join(cube_dir, f))
        ]

        window[guiek_file_list].update(fnames)

        pixel = cube_data.read_pixel(row=int(cube_data.nrows / 2), col=int(cube_data.ncols / 2))

        fig = plt.figure(figsize=(5, 4), dpi=100)
        fig.add_subplot(111).plot(pixel)
        # Add the plot to the window
        draw_figure(window["-CANVAS-"].TKCanvas, fig)


window.close()

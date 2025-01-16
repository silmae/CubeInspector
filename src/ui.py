
import PySimpleGUI as sg
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#### Plotting constants ####

axis_label_font_size = 16
"""Axis label font size"""
tick_label_font_size = 14
"""Tick label font size"""
save_resolution = 300
"""Save resolution for plots in dots per inch."""

false_color_base_width = 7
false_color_base_height = 6

monitor_dpi = 96

# Set Matplotlib to use TKinter backend apparently?
matplotlib.use("TkAgg")

# Set a nice color theme
sg.theme('dark grey 9')

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
guiek_save_figures = "-SAVE FIGURES-"
guiek_clear_button = "-CLEAR-"
# Pixel plot component handles
guiek_spectral_min_input = "-SPECTRAL MIN INPUT-"
guiek_spectral_max_input = "-SPECTRAL MAX INPUT-"
guiek_spectal_clip_min_wl_text = "-SPECTRAL CLIP MIN TEXT WL-"
guiek_spectal_clip_max_wl_text = "-SPECTRAL CLIP MAX TEXT WL-"
guiek_spectral_clip_button = "-SPECTRAL CLIP-"
guiek_ylim_input = "-YLIM INPUT-"
guiek_ylim_apply_button = "-YLIM APPLY-"


def initialize_ui():
    fig_px_plot = plt.figure(figsize=(5, 4), dpi=100)
    ax_px_plot = fig_px_plot.add_subplot(111)

    # Pixel plot
    ax_px_plot.xaxis.set_tick_params(labelsize=tick_label_font_size)
    ax_px_plot.yaxis.set_tick_params(labelsize=tick_label_font_size)

    # False color imshow
    fig_false_color, ax_false_color = get_false_color_fig_and_ax(false_color_base_width, false_color_base_height)

    # Layout stuff #############

    multiline_size = (50,15)

    frame_layout_cube_meta = [
        [sg.Multiline(size=multiline_size, key=guiek_cube_meta_text), ]
    ]
    frame_layout_ouput = [
        [sg.Multiline(size=multiline_size, reroute_stdout=True, k=guiek_console_output, autoscroll=True, horizontal_scroll=True), ]
    ]

    cube_meta_column = [
        [sg.Frame("Cube metadata", frame_layout_cube_meta, expand_x=True, expand_y=True)],
        [sg.Frame("Output", frame_layout_ouput, expand_x=True, expand_y=True)]
    ]


    #cube_meta_column = [
        #[sg.Push(),sg.Text("Cube metadata"),sg.Push()],
        #[sg.Multiline(size=multiline_size, key=guiek_cube_meta_text),],
        #[sg.Push(),sg.Text("Output"),sg.Push()],
        #[sg.Multiline(size=multiline_size, reroute_stdout=True, k=guiek_console_output, autoscroll=True,horizontal_scroll=True),],
    #]


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
            sg.Button("Clip", key=guiek_spectral_clip_button, enable_events=True),
            sg.Push(),
            sg.In(size=(7,1), key=guiek_ylim_input, tooltip="Y axis limit."),
            sg.Button("Ylim", key=guiek_ylim_apply_button, enable_events=True)
        ]
    ]

    layout = [
        [
            sg.Column(cube_meta_column, expand_x=True, expand_y=True),
            sg.VSeperator(),
            sg.Column(cube_column, expand_x=True, expand_y=True),
            sg.VSeperator(),
            sg.Column(pixel_plot_column,  pad=(0, 100), expand_x=True, expand_y=True),
        ],
        [sg.HSeparator()],
        [
            sg.Push(),
            sg.Button("Save Cube", key=guiek_save_cube, enable_events=True, disabled=True,
                      tooltip="Saves the main cube as reflectance cube (with .dat extension). Only available \n"
                              "after dark and white corrections are calculated. Not available for precomputed \n "
                              "reflectance cubes."),

            sg.Button("Save Figures", key=guiek_save_figures, enable_events=True, disabled=True,
                      tooltip="Saves figures."),
        ],
        [sg.VPush()],
    ]

    window = sg.Window("Cube Inspector", layout=layout, margins=(50,50), finalize=True, resizable=True)
    window.set_min_size((500,300))
    window[guiek_console_output].Widget.configure(wrap='none')

    # Bind Enter key(s) to input fields that can use it

    window[guiek_r_input].bind("<Return>", "_Enter")
    window[guiek_r_input].bind("<Return>", "KP_Enter")
    window[guiek_g_input].bind("<Return>", "_Enter")
    window[guiek_g_input].bind("<Return>", "KP_Enter")
    window[guiek_b_input].bind("<Return>", "_Enter")
    window[guiek_b_input].bind("<Return>", "KP_Enter")
    window[guiek_spectral_min_input].bind("<Return>", "_Enter")
    window[guiek_spectral_min_input].bind("<Return>", "KP_Enter")
    window[guiek_spectral_max_input].bind("<Return>", "_Enter")
    window[guiek_spectral_max_input].bind("<Return>", "KP_Enter")
    window[guiek_ylim_input].bind("<Return>", "_Enter")
    window[guiek_ylim_input].bind("<Return>", "KP_Enter")

    #resize canvases, cube metadata and output
    """
    We don't need the for loops here as they are single elements. 
    Should figure out a way to resize without losing the buttons. And 
    maybe set a minimum size for the window. Good work! We continue from 
    here. - Kimmo 
    """
    # window[guiek_cube_false_color].expand(expand_x=True, expand_y=True)
    # window[guiek_pixel_plot_canvas].expand(expand_x=True, expand_y=True)
    # window[guiek_cube_meta_text].expand(expand_x=True, expand_y=True)
    # window[guiek_console_output].expand(expand_x=True, expand_y=True)

    # There if some flickering when clicking on the RGB image.
    # Also, I think the pixel plot size should not change with window size as it will be saved as-is.
    window.Maximize()

    return window, fig_px_plot, fig_false_color, ax_px_plot, ax_false_color


def get_false_color_canvas_size(aspect_ratio, runtime_state):
    """Get the size of the false color canvas according to the aspect ratio of the cube and the size of the window."""

    w_size = runtime_state['window'].size
    window_width_inch = w_size[0]/monitor_dpi
    window_height_inch = w_size[1]/monitor_dpi
    canvas_width_inch = window_width_inch / 3.5
    canvas_height_inch = window_height_inch / 3

    # print(f"Window width in px = {w_size[0]} and in inches = {w_size[0]/monitor_dpi}")
    # print(f"Window height in px = {w_size[1]} and in inches = {w_size[1] / monitor_dpi}")

    if aspect_ratio >= 1:
        new_height = int(canvas_width_inch / aspect_ratio)
        new_width = canvas_width_inch
    else:
        new_width = int(canvas_height_inch * aspect_ratio)
        new_height = canvas_height_inch

    return new_width, new_height


def get_false_color_fig_and_ax(w,h):
    fig_false_color = plt.figure(figsize=(w, h), dpi=100)
    ax_false_color = fig_false_color.add_subplot(111)

    ax_false_color.xaxis.set_tick_params(labelsize=tick_label_font_size)
    ax_false_color.yaxis.set_tick_params(labelsize=tick_label_font_size)

    return fig_false_color, ax_false_color


def draw_figure(canvas, figure):
    """Helper function to draw things on canvas."""

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def update_UI_component_state(RUNTIME):
    """Updates the disabled state of all UI buttons and RGB inboxes."""

    window = RUNTIME['window']

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
    window[guiek_save_figures].update(disabled=True)

    if RUNTIME['img_array'] is not None:
        window[guiek_cube_show_button].update(disabled=False)

        if not RUNTIME['cube_is_reflectance']:
            window[guiek_dark_file_browse].update(disabled=False)
            window[guiek_white_file_browse].update(disabled=False)

    # We only need calculation stuff is loaded cube is not already reflectance
    if not RUNTIME['cube_is_reflectance']:

        if RUNTIME['img_array_dark'] is not None:
            window[guiek_dark_show_button].update(disabled=False)
            if not RUNTIME['dark_corrected'] and RUNTIME['dark_median'] is not None:
                window[guiek_calc_dark].update(disabled=False)
            else:
                window[guiek_calc_dark].update(disabled=True)
        if RUNTIME['img_array_white'] is not None:
            window[guiek_white_show_button].update(disabled=False)
            window[guiek_white_select_region].update(disabled=False)
            window[guiek_white_select_whole].update(disabled=False)
            if not RUNTIME['white_corrected'] and RUNTIME['white_spectra'] is not None and RUNTIME['dark_corrected']:
                window[guiek_calc_white].update(disabled=False)
            else:
                window[guiek_calc_white].update(disabled=True)

        if RUNTIME['white_corrected'] or RUNTIME['dark_corrected']:
            window[guiek_save_cube].update(disabled=False)

    if RUNTIME['cube_dir_path'] is not None:
        window[guiek_save_figures].update(disabled=False)

    # Update the RGB inboxes as well
    window[guiek_r_input].update(str(RUNTIME['band_R']))
    window[guiek_g_input].update(str(RUNTIME['band_G']))
    window[guiek_b_input].update(str(RUNTIME['band_B']))

    window[guiek_spectral_min_input].update(str(RUNTIME['spectral_clip_min']))
    window[guiek_spectral_max_input].update(str(RUNTIME['spectral_clip_max']))

"""

Some resources that might be useful:

Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/
GitHub project PySimpleGUI: https://github.com/PySimpleGUI/PySimpleGUI

Refreshing plot: https://gist.github.com/KenoLeon/e913de9e1fe690ebe287e6d1e54e3b97
"""

import math

import spectral as spy
import numpy as np
from matplotlib import ticker
from matplotlib.patches import Rectangle

from src.cube_handling import find_cube, calc_dark, calc_white, cube_meta, open_cube
from src.ui import *
from src.state import state_load, state_save, get_runtime_state, update_runtime_ui_components, get_save_state
from src.utils import get_base_name_wo_postfix, img_array_to_rgb, infer_runtime_RGB_value, cube_dimensions

# TODO Incorporate dark and white selection logic with buttons.
# TODO Show corresponding wl for RGB band selection. In plot or as a text box?
# TODO Gaussian or mean RGB.



_SETTINGS = {
    'drag_threshold': 5
}

def mouse_click_event(eventi):
    """Handle mouse click event (button 1 down)."""

    # Store data in case there will be a drag
    runtime_state['rect_x_0'] = eventi.xdata
    runtime_state['rect_y_0'] = eventi.ydata

def mouse_release_event(eventi):
    """Handles Matplotlib mouse button release event.

    Note that you cannot raise PySimpleGUI popups in here as it will block the event loop.

    :param eventi:
        Matplotlib mouse button release event.
    """

    # Left click for right-handed mouse.
    if eventi.button == 1:
        x0 = runtime_state['rect_x_0']
        y0 = runtime_state['rect_y_0']
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

            # print(f"Mouse drag from ({int(drag_start_x)},{int(drag_start_y)}) to ({int(drag_end_x)},{int(drag_end_y)}).")

            rows = list(np.arange(start=drag_start_x, stop=drag_end_x, step=1))
            cols = list(np.arange(start=drag_start_y, stop=drag_end_y, step=1))

            if runtime_state['view_mode'] == 'cube':
                sub_image = runtime_state['img_array'][cols][:, rows]
            elif runtime_state['view_mode'] == 'dark':
                sub_image = runtime_state['img_array_dark'][cols][:, rows]
            elif runtime_state['view_mode'] == 'white':
                sub_image = runtime_state['img_array_white'][cols][:, rows]
            else:
                print(f"WARNING: View mode '{runtime_state['view_mode']}' not supported.")

            sub_mean = np.mean(sub_image, axis=(0,1))
            sub_std = np.std(sub_image, axis=(0,1))

            update_px_rgb_lines()
            update_px_plot(spectrum=sub_mean, std=sub_std, x0=drag_start_x, y0=drag_start_y, x1=drag_end_x, y1=drag_end_y)

            if runtime_state['selecting_white']:
                runtime_state['white_spectra'] = sub_mean
                runtime_state['selecting_white'] = False
                print(f"White spectrum saved.")

        else:
            # Else it was just a click and we can reset x0 and y0
            runtime_state['rect_x_0'] = None
            runtime_state['rect_y_0'] = None
            pixel = runtime_state['img_array'][int(y), int(x)]
            # print(f"Mouse click at ({int(x)},{int(y)}).")
            update_px_rgb_lines()
            update_px_plot(spectrum=pixel, x0=x, y0=y)

        update_UI_component_state(RUNTIME=runtime_state)


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

    if runtime_state['cube_data'] is None:
        raise RuntimeError(f"No cube selected: I will not set clipping for you.")

    bands = runtime_state['cube_bands']
    actual_min = np.clip(min_int, bands[0], bands[-1])
    actual_max = np.clip(max_int, bands[0], bands[-1])
    runtime_state['spectral_clip_min'] = actual_min
    runtime_state['spectral_clip_max'] = actual_max


def update_px_rgb_lines():
    """Update the vertical line positions for pixel plot to the _RUNTIME.

    Additionally, updates the wavelength textblocks of selected bands.

    NOTE
    You should call update_px_plot() after calling this for the actual update on screen.
    """

    for handle in runtime_state['rgb_handles']:
        handle.remove()

    runtime_state['rgb_handles'] = []
    band_keys = ['band_R', 'band_G', 'band_B']
    line_colors = ['red', 'green', 'blue']

    for i in range(3):
        should_fill, value = infer_runtime_RGB_value(runtime_state[band_keys[i]])
        if not should_fill:
            runtime_state['rgb_handles'].append(ax_px_plot.axvline(x=value, color=line_colors[i]))

    update_band_wl_textblocks()


def update_band_wl_textblocks():

    wl_textblock_keys = [guiek_r_wl_text, guiek_g_wl_text, guiek_b_wl_text]
    band_keys = ['band_R', 'band_G', 'band_B']
    for i in range(3):
        should_fill, value = infer_runtime_RGB_value(runtime_state[band_keys[i]])
        if not should_fill and runtime_state['cube_wls'] is not None:
            runtime_state['window'][wl_textblock_keys[i]].update(f"{runtime_state['cube_wls'][value]:.2f} nm")
        elif should_fill:
            runtime_state['window'][wl_textblock_keys[i]].update(f"---.-- nm")


def update_spectral_clip_wl_text():
    runtime_state['window'][guiek_spectal_clip_min_wl_text].update(f"{runtime_state['cube_wls'][runtime_state['spectral_clip_min']]:.2f} nm")
    runtime_state['window'][guiek_spectal_clip_max_wl_text].update(f"{runtime_state['cube_wls'][runtime_state['spectral_clip_max']]:.2f} nm")


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
    runtime_state['fig_agg_px_plot'].get_tk_widget().forget()
    ax_px_plot.set_xlim(runtime_state['spectral_clip_min'], runtime_state['spectral_clip_max'])
    update_spectral_clip_wl_text()

    plot_color = None
    ylim = runtime_state['px_plot_ylim']

    # Plot the plot and save it. Update
    if spectrum is not None:
        # Store the random color used in pixel plot to draw a rectangle over RGB image later.
        p = ax_px_plot.plot(spectrum)
        plot_color = p[-1].get_color()

        runtime_state['plots'].append(spectrum)
        new_plot_max = 0
        for plot in runtime_state['plots']:
            plot_max = np.max(plot)
            if plot_max > new_plot_max:
                new_plot_max = plot_max

        # print(f"Setting ylim to {_RUNTIME['plot_max']}")
        # Set ylim a little higher than the max value of any of the plots
        auto_ylim = new_plot_max * 1.05
        if ylim is None:
            runtime_state['px_plot_ylim_auto'] = auto_ylim
            ax_px_plot.set_ylim(0, auto_ylim)
        else:
            try:
                ax_px_plot.set_ylim(0, ylim)
            except:
                print("Could not set y-axis limit from user feed. Using automatic y-axis limit.")
                runtime_state['px_plot_ylim_auto'] = auto_ylim
                ax_px_plot.set_ylim(0, auto_ylim)

    # Even if new spectrum was not given, we can try to update ylim
    else:
        if ylim is not None:
            ax_px_plot.set_ylim(0, ylim)
        else:
            ax_px_plot.set_ylim(0, runtime_state['px_plot_ylim_auto'])

    bands = runtime_state['cube_bands']
    wls = runtime_state['cube_wls']

    if bands is not None and wls is not None and runtime_state['sec_axes_px_plot'] is None:

        # Hack to print into light form for HyperBlend
        # s_max = np.max(spectrum)
        # for i,wl in enumerate(wls):
        #     print(f"{wl:.4} {spectrum[i]/s_max:.9}")

        ax_px_plot.set_xlabel('Band', fontsize=axis_label_font_size)

        def forward(x):
            return np.interp(x, bands, wls)

        def inverse(x):
            return np.interp(x, wls, bands)

        # The ticks at the secondary axis are not perfect but good enough for now
        secax = ax_px_plot.secondary_xaxis('top', functions=(forward, inverse))
        secax.set_xticks(wls)
        secax.xaxis.set_tick_params(labelsize=tick_label_font_size, rotation=70)
        secax.set_xlabel(r"Wavelength [$nm$]", fontsize=axis_label_font_size)
        runtime_state['sec_axes_px_plot'] = secax

    if std is not None:
        ax_px_plot.fill_between(runtime_state['cube_bands'], spectrum - (std / 2), spectrum + (std / 2), alpha=0.2)

    runtime_state['fig_agg_px_plot'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

    # Draw rectangle over RGB canvas and refresh
    if std is not None and x0 is not None and y0 is not None and x1 is not None and y1 is not None:
        width = math.fabs(x0 - x1)
        height = math.fabs(y0 - y1)
        runtime_state['fig_agg_false_color'].get_tk_widget().forget()
        if plot_color is not None:
            handle = ax_false_color.add_patch(Rectangle((x0, y0), width=width, height=height, fill=False, edgecolor=plot_color))
        else:
            handle = ax_false_color.add_patch(Rectangle((x0, y0), width=width, height=height, fill=False, edgecolor='gray'))
        runtime_state['rectangle_handles'].append(handle)
        runtime_state['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

    # Draw click location as a dot
    elif x0 is not None and y0 is not None:
        runtime_state['fig_agg_false_color'].get_tk_widget().forget()
        handle = ax_false_color.scatter(int(x0), int(y0))
        runtime_state['dot_handles'].append(handle)
        runtime_state['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def update_false_color_canvas():
    """Updates the false color canvas based on _RUNTIME globals.

    Uses user-defined bands for RGB reconstruction. Pixel values are automatically
    scaled for the whole possible range.

    Check if RGB exists in _RUNTIME and use that
    _RUNTIME RGB has to be loaded from _STATE
    If not, get RGB from metadata.
    """

    if runtime_state['cube_data'] is None:
        print(f"Cube not set. Nothing to clear.")
        return

    possibly_R_str = str(runtime_state['band_R'])
    possibly_G_str = str(runtime_state['band_G'])
    possibly_B_str = str(runtime_state['band_B'])
    meta_bands = [int(band) - 1 for band in runtime_state['cube_data'].metadata['default bands']]

    if len(possibly_R_str) == 0: # empty string
        possibly_R_str = str(meta_bands[0])
        runtime_state['band_R'] = possibly_R_str
    if len(possibly_G_str) == 0: # empty string
        possibly_G_str = str(meta_bands[1])
        runtime_state['band_G'] = possibly_G_str
    if len(possibly_B_str) == 0: # empty string
        possibly_B_str = str(meta_bands[2])
        runtime_state['band_B'] = possibly_B_str

    view_mode = runtime_state['view_mode']

    if view_mode == 'cube' and not runtime_state['selecting_white']:

        if runtime_state['img_array'] is None:
            print(f"Image array None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(runtime_state['img_array'], possibly_R_str, possibly_G_str, possibly_B_str)

    elif view_mode == 'dark':

        if runtime_state['img_array_dark'] is None:
            print(f"Image array for dark is None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(runtime_state['img_array_dark'], possibly_R_str, possibly_G_str, possibly_B_str)

    elif view_mode == 'white' or runtime_state['selecting_white']:

        if runtime_state['img_array_white'] is None:
            print(f"Image array for white is None. Nothing to show.")
            return

        false_color_rgb = img_array_to_rgb(runtime_state['img_array_white'], possibly_R_str, possibly_G_str, possibly_B_str)
    else:
        print(f"WARNING: unknown view mode '{view_mode}' and/or selection combination selecting "
              f"white={runtime_state['selecting_white']}.")
        return

    runtime_state['fig_agg_false_color'].get_tk_widget().forget()
    # Clear axis object because if the next image is of different size,
    # it will break pixel indexing for mouse selection

    w, h, ar = cube_dimensions(runtime_state['cube_data'])

    new_canvas_w, new_canvas_h = get_false_color_canvas_size(aspect_ratio=ar, runtime_state=runtime_state)
    if new_canvas_w != runtime_state['false_color_canvas_width'] or new_canvas_h != runtime_state['false_color_canvas_height']:
        runtime_state['false_color_canvas_width'] = new_canvas_w
        runtime_state['false_color_canvas_height'] = new_canvas_h
        fig_false_color.set_size_inches(new_canvas_w, new_canvas_h, forward=True)

    ax_false_color.cla()
    ax_false_color.imshow(false_color_rgb)
    ax_false_color.set_xlabel('Samples', fontsize=axis_label_font_size)
    ax_false_color.set_ylabel('Lines', fontsize=axis_label_font_size)
    runtime_state['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)


def clear_plot():
    """Clear pixel plot.

    TODO maybe make more general to clear any pyplot axis?
    """

    print(f"Clearing pixel plot")
    runtime_state['fig_agg_px_plot'].get_tk_widget().forget()
    ax_px_plot.clear()
    runtime_state['fig_agg_px_plot'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)

    print(f"Clearing false color RGB")
    # Remove rectangles and other Artists
    # while ax_false_color.patches:
    #     ax_false_color.patches[0].remove()
    for handle in runtime_state['rectangle_handles']:
        handle.remove()
    for handle in runtime_state['dot_handles']:
        handle.remove()
    for handle in runtime_state['rgb_handles']:
        handle.remove()

    runtime_state['rectangle_handles'] = []
    runtime_state['dot_handles'] = []
    runtime_state['rgb_handles'] = []
    runtime_state['sec_axes_px_plot'] = None
    runtime_state['plots'] = []
    update_false_color_canvas()


def restore_from_previous_session():
    """Partially restores UI to the state it was when last closed.

    Loads all the cubes but does not do any corrections.
    """

    print(f"Trying to restore state from previous session.")

    save_state = get_save_state()
    if save_state['main_cube_hdr_path'] is not None:
        path = save_state['main_cube_hdr_path']
        window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_cube_file_selected(path)
    if save_state['dark_cube_hdr_path'] is not None:
        path = save_state['dark_cube_hdr_path']
        window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_dark_file_selected(path)
    if save_state['white_cube_hdr_path'] is not None:
        path = save_state['white_cube_hdr_path']
        window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(path))
        handle_white_file_selected(path)

    runtime_state['view_mode'] = 'cube'
    update_false_color_canvas()


def handle_cube_file_selected(file_path:str):

    # Reset in case a new cube is selected
    runtime_state['selecting_white'] = False
    runtime_state['view_mode'] = 'cube'

    runtime_state['dark_median'] = None
    runtime_state['img_array_dark'] = None
    runtime_state['dark_corrected'] = False

    runtime_state['white_spectra'] = None
    runtime_state['img_array_white'] = None
    runtime_state['white_corrected'] = False

    try_to_open_cube(file_path, mode='cube')
    update_false_color_canvas()
    update_UI_component_state(RUNTIME=runtime_state)


def try_to_open_cube(path: str, mode: str):

    can_open, hdr_path, raw_path = find_cube(path, mode=mode, save_state=get_save_state(), runtime_state=runtime_state)

    if can_open:

        open_cube(hdr_path=hdr_path, data_path=raw_path, mode=mode, runtime_state=runtime_state)

        if mode == 'cube':
            clear_plot()
            cube_meta(window[guiek_cube_meta_text], runtime_state=runtime_state)

        # Draw cube to canvas
        update_false_color_canvas()
        update_band_wl_textblocks()
        update_spectral_clip_wl_text()

def handle_dark_file_selected(file_path: str):

    runtime_state['view_mode'] = 'dark'

    try_to_open_cube(file_path, mode='dark')
    # image_array_dark should now have a value
    dark_cube = runtime_state['img_array_dark']
    print(f"Dark cube set. Calculating median of the scan lines. WAIT a bit, please.")

    # Scan lines are on axis=0
    med = np.median(dark_cube, axis=0)
    # print(f"DEBUG: dark median shape: {med.shape}")
    runtime_state['dark_median'] = med

    print(f"Dark median saved.")
    update_false_color_canvas()
    update_UI_component_state(RUNTIME=runtime_state)


def handle_white_file_selected(file_path: str):
    runtime_state['selecting_white'] = True
    runtime_state['view_mode'] = 'white'
    try_to_open_cube(file_path, mode='white')
    runtime_state['selecting_white'] = False

    print(f"White cube set.")

    update_false_color_canvas()
    update_UI_component_state(RUNTIME=runtime_state)


def save_reflectance_cube():
    save_state = get_save_state()
    basepath = str(save_state['main_cube_hdr_path']).rsplit('.', maxsplit=1)[0]
    if runtime_state['white_corrected']:
        save_hdr_path = f"{basepath}_CI_reflectance.hdr"
    elif runtime_state['dark_corrected']:
        save_hdr_path = f"{basepath}_CI_darkcorrected.hdr"
    else:
        raise RuntimeError(f"Cannot save cube: runtime state not recognized.")
    print(f"Trying to save cube to '{save_hdr_path}'.")
    spy.envi.save_image(hdr_file=save_hdr_path, image=runtime_state['img_array'], dtype=np.float32, ext='.dat', metadata=runtime_state['cube_data'].metadata)
    print(f"Cube saved.")


def save_figures():
    """Saves false color RGB image and pixel plot to disk to same path where the cubes are in."""

    path_save_rgb = runtime_state['cube_dir_path'] + '/' + 'false_rgb.png'
    path_save_px_plot = runtime_state['cube_dir_path'] + '/' + 'pixel_plot.png'

    # Pyplot reference figures by index number so this way we can save them separately.
    plt.figure(1)
    plt.savefig(path_save_px_plot, dpi=save_resolution, bbox_inches='tight', transparent=False)
    print(f"Saved false color RGB image to '{path_save_rgb}'.")

    plt.figure(2)
    plt.savefig(path_save_rgb, dpi=save_resolution, bbox_inches='tight', transparent=False)
    print(f"Saved pixel plot to '{path_save_px_plot}'.")


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

    cid_press = fig_false_color.canvas.mpl_connect('button_press_event', mouse_click_event)
    cid_release = fig_false_color.canvas.mpl_connect('button_release_event', mouse_release_event)

    # """Enclose the whole GUI loop to try-except block to catch any errors and save state before closing down.
    # This should be used only for executable as it makes debugging harder. """
    # try:
    # Infinite GUI loop
    while True:
        event, values = window.read()

        # Clutters console but can be useful for debugging.
        # print(f"event {event}, values: {values}")

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        elif event == guiek_clear_button:
            clear_plot()

        elif (event == guiek_spectral_clip_button or
              event == guiek_spectral_min_input + "_Enter" or event == guiek_spectral_min_input + "KP_Enter" or
              event == guiek_spectral_max_input + "_Enter" or event == guiek_spectral_max_input + "KP_Enter"):
            print("or enter")
            try:
                infer_runtime_spectral_clip(values[guiek_spectral_min_input], values[guiek_spectral_max_input])
                update_px_plot()
            except Exception as e:
                print(f"Could not set clip value: \n {e}")

        elif event == guiek_ylim_apply_button or event == guiek_ylim_input + "_Enter" or event == guiek_ylim_input + "KP_Enter":
            feed = values[guiek_ylim_input]
            if len(feed) == 0:
                print("empty ylim input. setting it to none")
                runtime_state['px_plot_ylim'] = None
                update_px_plot()
            else:
                try:
                    feed_int = float(feed)
                    runtime_state['px_plot_ylim'] = feed_int
                    update_px_plot()
                except:
                    print(f"Could not cast y-axis limit input '{feed}' into a float. Ignoring input.")

        elif event == guiek_cube_file_browse:
            window[guiek_cube_show_filename].update(value=get_base_name_wo_postfix(values[guiek_cube_file_browse]))
            handle_cube_file_selected(values[guiek_cube_file_browse])

        elif event == guiek_dark_file_browse:
            window[guiek_dark_show_filename].update(value=get_base_name_wo_postfix(values[guiek_dark_file_browse]))
            handle_dark_file_selected(values[guiek_dark_file_browse])

        elif event == guiek_white_file_browse:
            window[guiek_white_show_filename].update(value=get_base_name_wo_postfix(values[guiek_white_file_browse]))
            handle_white_file_selected(values[guiek_white_file_browse])

        elif event == guiek_cube_show_button:
            runtime_state['view_mode'] = 'cube'
            update_false_color_canvas()

        elif event == guiek_dark_show_button:
            runtime_state['view_mode'] = 'dark'
            update_false_color_canvas()

        elif event == guiek_white_show_button:
            runtime_state['view_mode'] = 'white'
            update_false_color_canvas()

        elif event == guiek_white_select_region:
            runtime_state['selecting_white'] = True
            print(f"White region selection is now on. Drag across the image to select an "
                  f"area which will be used as a white reference.")

        elif event == guiek_white_select_whole:
            white_array = runtime_state['img_array_white']
            white_mean = np.mean(white_array, axis=(0,1))
            runtime_state['white_spectra'] = white_mean
            print(f"White reference spectra set. You can now use the Calculate button to "
                  f"calculate reflectance.")

        elif event == guiek_calc_dark:
            calc_dark(runtime_state=runtime_state)
            update_false_color_canvas()

        elif event == guiek_calc_white:
            calc_white(runtime_state=runtime_state)
            update_false_color_canvas()

        elif event == guiek_save_cube:
            save_reflectance_cube()

        elif event == guiek_save_figures:
            save_figures()

        elif (event == guiek_rgb_update_button or
              event == guiek_r_input + "_Enter" or event == guiek_r_input + "KP_Enter" or
              event == guiek_g_input + "_Enter" or event == guiek_g_input + "KP_Enter" or
              event == guiek_b_input + "_Enter" or event == guiek_b_input + "KP_Enter"):
            try:
                # Just try casting before continuing
                _, _ = infer_runtime_RGB_value(values[guiek_r_input])
                _, _ = infer_runtime_RGB_value(values[guiek_g_input])
                _, _ = infer_runtime_RGB_value(values[guiek_b_input])
                runtime_state['band_R'] = values[guiek_r_input] #int(values[guiek_r_input])
                runtime_state['band_G'] = values[guiek_g_input] #int(values[guiek_g_input])
                runtime_state['band_B'] = values[guiek_b_input] #int(values[guiek_b_input])
                update_px_rgb_lines()
                update_px_plot()
                update_false_color_canvas()
            except ValueError as ve:
                print(f"WARNING: Failed casting band to an integer. False color image not updated.")

        else:
            print("We should not have arrived in here in the main loop iffing.")

        # Update UI after every event is handled.
        update_UI_component_state(RUNTIME=runtime_state)

    # Use this for executable
    # except Exception as e:
    #     sg.popup_error(f"An error occurred. Your state will be saved before closing down. \n"
    #                    f"Error message:\n {e}")

    state_save()
    window.close()


if __name__ == '__main__':

    # Set Spectral Python library to support non-lowercase file names
    spy.settings.envi_support_nonlowercase_params = True

    window, fig_px_plot, fig_false_color, ax_px_plot, ax_false_color = initialize_ui()
    update_runtime_ui_components(window, fig_px_plot, fig_false_color, guiek_pixel_plot_canvas, guiek_cube_false_color)

    # Get the runtime state dictionary only once
    runtime_state = get_runtime_state()
    main()

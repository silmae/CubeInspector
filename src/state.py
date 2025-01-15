
import os

from src import toml_handling as TH
from ui import draw_figure

# State dict save directory and filename
state_dir_path = os.getcwd()
state_file_name = "ci_state"

def get_runtime_state():
    return _RUNTIME_STATE

def get_save_state():
    return _SAVE_STATE

def update_runtime_ui_components(window, fig_px_plot, fig_false_color, guiek_pixel_plot_canvas, guiek_cube_false_color):
    _RUNTIME_STATE['window'] = window
    _RUNTIME_STATE['fig_agg_px_plot'] = draw_figure(window[guiek_pixel_plot_canvas].TKCanvas, fig_px_plot)
    _RUNTIME_STATE['fig_agg_false_color'] = draw_figure(window[guiek_cube_false_color].TKCanvas, fig_false_color)

# Keep most of the global stuff in this single dictionary for later access
_RUNTIME_STATE = {
    'window': None,
    'fig_agg_px_plot': None,
    'sec_axes_px_plot': None,
    'fig_agg_false_color': None,
    'pltFig': False,

    'cube_dir_path': None,
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

    'px_plot_ylim': None, # float or None. None is interperted so that automatic scaling is used
    'px_plot_ylim_auto': None, # used if user overrides ylim at some point but want to go back to previous automatic lim

    'plots': [],
}


_SAVE_STATE = {
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


def state_load():
    """Load state from disk and sets _STATE global dict.

    Also sets RGB bands to _RUNTIME global dict.
    """

    loaded_state = TH.read_toml_as_dict(directory=state_dir_path, filename=state_file_name)
    for key, value in loaded_state.items():
        _SAVE_STATE[key] = value

    _RUNTIME_STATE['band_R'] = str(_SAVE_STATE['band_R'])
    _RUNTIME_STATE['band_G'] = str(_SAVE_STATE['band_G'])
    _RUNTIME_STATE['band_B'] = str(_SAVE_STATE['band_B'])

    print(f"Previous state loaded.")


def state_save():
    """Save the _STATE dict to disk.
    """

    _SAVE_STATE['band_R'] = str(_RUNTIME_STATE['band_R'])
    _SAVE_STATE['band_G'] = str(_RUNTIME_STATE['band_G'])
    _SAVE_STATE['band_B'] = str(_RUNTIME_STATE['band_B'])

    TH.write_dict_as_toml(dictionary=_SAVE_STATE, directory=state_dir_path, filename=state_file_name)
    print(f"State saved")

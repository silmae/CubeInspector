
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import math

"""
TODO kirjota kokonaan uudestaan. Tää on jotenki ihan päin helvettiä.
"""

val_measurement_s1_v_13_dL = {
    'hdr_path': "D:/Koodi/Python/HyperBlend/Algae spectral images/Algae_imaging_Algae_1_1.3_L_2023-11-30_12-22-04/capture\\Algae_imaging_Algae_1_1.3_L_2023-11-30_12-22-04_CI_reflectance.hdr",
    'data_path': "D:/Koodi/Python/HyperBlend/Algae spectral images/Algae_imaging_Algae_1_1.3_L_2023-11-30_12-22-04/capture\\Algae_imaging_Algae_1_1.3_L_2023-11-30_12-22-04_CI_reflectance.dat",
    'x0': 500,
    'y0': 170,
    'x1': 502,
    'y1': 632,
    'band_b': 40,
    'band_r': 190,
}
# Mouse drag from (497,170) to (542,632).

# x- suuntaan Mouse drag from (370,378) to (672,441).

val_simulated_s1_v_13_dL = {
    'hdr_path': "D:/Koodi/Python/HyperBlend/HyperBlend/scenes/scene_validation_algae_1_v_1.3L/cube\\reflectance_cube_validation_algae_1_v_1.3L_CI_reflectance.hdr",
    'data_path': "D:/Koodi/Python/HyperBlend/HyperBlend/scenes/scene_validation_algae_1_v_1.3L/cube\\reflectance_cube_validation_algae_1_v_1.3L_CI_reflectance.dat",
    'x0': 200,
    'y0': 75,
    'x1': 202,
    'y1': 320,
    'band_b': 40,
    'band_r': 250,
}


def get_diff_box(dict_sel: dict, swap=False):
    cube_data = envi.open(file=dict_sel['hdr_path'], image=dict_sel['data_path'])
    img_array = cube_data.load().asarray()
    if swap:
        img_array = np.swapaxes(img_array, axis1=0, axis2=1)
    x0 = dict_sel['x0']
    x1 = dict_sel['x1']
    y0 = dict_sel['y0']
    y1 = dict_sel['y1']
    band_b = dict_sel['band_b']
    band_r = dict_sel['band_r']
    r = 1
    # sub_image_b = np.mean(img_array[x0:x1,y0:y1,band_b-r:band_b+r], axis=2)
    sub_image_b = img_array[x0:x1,y0:y1,band_b]
    # plt.plot(np.mean(sub_image_b, axis=1))
    # plt.show()
    # sub_image_r = np.mean(img_array[x0:x1,y0:y1,band_r-r:band_r+r], axis=2)
    sub_image_r = img_array[x0:x1,y0:y1,band_r]
    sub_image = np.stack((sub_image_b,sub_image_r), axis=2)
    sub_image_max = np.max(sub_image)
    sub_image_mean = np.mean(sub_image, axis=0)
    sub_image_mean = sub_image_mean / sub_image_mean.max()
    sub_image_var = np.var(sub_image, axis=0)
    sub_image_var = sub_image_var / sub_image_var.max()
    x_axis = range(int(math.fabs(y0-y1)))
    return sub_image_mean, sub_image_var, x_axis


def thing():
    sub_image_mean, sub_image_var, x_sim = get_diff_box(val_simulated_s1_v_13_dL, swap=True)

    for i in range(2):
        mean = sub_image_mean[:,i]
        var = sub_image_var[:,i]

        if i == 0:
            plot_label = 'blue simulated'
            color = 'blue'
            ls = 'solid'
        else:
            plot_label = 'red simulated'
            color = 'red'
            ls = 'solid'

        plt.plot(mean, label=plot_label, color=color, linestyle=ls)
        # plt.fill_between(x_axis, mean - (var / 2), mean + (var / 2), alpha=0.2)

    sub_image_mean, sub_image_var, x_axis = get_diff_box(val_measurement_s1_v_13_dL)

    for i in range(2):
        mean = sub_image_mean[:, i]
        var = sub_image_var[:, i]
        mean = np.interp(x_sim, x_axis, mean)

        if i == 0:
            plot_label = 'blue measured'
            color = 'blue'
            ls = 'dashed'
        else:
            plot_label = 'red measured'
            color = 'red'
            ls = 'dashed'

        plt.plot(mean, label=plot_label, color=color, linestyle=ls)
        # plt.fill_between(x_axis, mean - (var / 2), mean + (var / 2), alpha=0.2)

    plt.legend()
    plt.show()



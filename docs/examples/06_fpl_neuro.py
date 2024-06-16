# -*- coding: utf-8 -*-

"""
# Neuroscience using `fastplotlib` and `pynapple`

This notebook will build up a complex visualization using `fastplotlib`, in conjunction with `pynapple`, to show how
`fastplotlib` can be a powerful tool in analysis and visualization of neural data!

"""

# %%

import warnings
warnings.simplefilter('ignore')

# %%

# uncomment and install if needed
# ! pip install scikit-image

# %%

import workshop_utils

import fastplotlib as fpl
import pynapple as nap
import numpy as np
from ipywidgets import IntSlider, Layout, VBox, HBox, FloatSlider
from skimage import measure
from sidecar import Sidecar

# %%

import warnings
warnings.simplefilter('ignore')

# %%
#
# ## Load the data
#
# **Recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded is the
# postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head
# is pointing in a specific direction.**

# get path to file
path = ""

# load data using pynapple
data = nap.load_file(path)

# %%

print(data)

# %%
#
# ## View the behavior and calcium data
#
# Hopefully, by the end of the summer we will have developed a tool ([`pynaviz`](
# https://github.com/pynapple-org/pynaviz)) that makes these visualizations and synchronizations even easier :D

# behavior shape
behavior_data = data["beh_video"]
print(behavior_data.shape)

# %%

# calcium shape
calcium_data = data["calcium_video"]
print(calcium_data.shape)

# %%
#
# **Minimize our view of the data to where both behavior and position data are available:**

frame_min = data["position_time_support"]["start"][0]
frame_max = data["position_time_support"]["end"][0]
print((frame_min, frame_max))

# %%
# #### Create a plot for calcium and behavior video

nap_figure = fpl.Figure(shape=(1,2), names=[["raw", "behavior"]])

nap_figure["raw"].add_image(data=calcium_data[0], name="raw_frame", cmap="viridis")
nap_figure["behavior"].add_image(data=behavior_data[0], cmap="gray")

# %%
# #### Create a slider that updates the behavior and calcium videos using `pyanapple`

# This time will be in milliseconds
synced_time = FloatSlider(min=frame_min, max=frame_max, step=0.1, description="s", layout=Layout(width="60%"))


def update_time(change):
    # get the index of synced slider
    time_s = change["new"]
    # get the corresponding calcium frame from the pynapple tensor
    frame_raw = calcium_data.get(time_s, time_units="s")
    # update the data in the plot
    nap_figure["raw"].graphics[0].data = frame_raw
    # get the corresponding behavior frame from the pynapple tensor
    frame_behavior = behavior_data.get(time_s, time_units="s")
    # update the data in the plot
    nap_figure["behavior"].graphics[0].data = frame_behavior


synced_time.observe(update_time, "value")

# %%
# **Here we are going to use `sidecar` to organize our visualizations better :D**

sc = Sidecar()
with sc:
    display(VBox([nap_figure.show(), synced_time]))

# %%

# manually set the vmin/vmax of the calcium data
nap_figure["raw"]["raw_frame"].vmax = 205
nap_figure["raw"]["raw_frame"].vmin = 25

# %%
#
# #### Calculate the spatial contours and overlay them on the raw calcium data

# get the masks
contour_masks = data.nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']['image_mask'].data[:]
# reshape the masks into a list of 105 components
contour_masks = list(contour_masks.reshape((len(contour_masks), 166, 136)))

# %%

# calculate each contour from the mask using `scikit-image.measure`
contours = list()

for mask in contour_masks:
    contours.append(np.vstack(measure.find_contours(mask)))

# %%
# #### Add the calculated contours as an overlay to the calcium video

contours_graphic = nap_figure["raw"].add_line_collection(data=contours, colors="w")

# %%
# **It is very easy to see that many of the identified neurons may be "bad" candidates. Let's remove them from the
# dataset as we go on in our anaylsis.**

# #### Select only head-direction neurons

# get the temporal data (calcium transients) from the nwb notebook
temporal_data = data["RoiResponseSeries"][:]
print(temporal_data)

# %%

# compute 1D tuning curved based on head angle
head_angle = data["ry"]

tuning_curves = nap.compute_1d_tuning_curves_continuous(temporal_data, head_angle, nb_bins = 120)

# %%
# #### Select the top 50 components

# select good components
good_ixs = list(np.argsort(np.ptp(tuning_curves, axis=0))[-50:])
bad_ixs = list(np.argsort(np.ptp(tuning_curves, axis=0))[:-50])

# %%
# #### Color the "good" and "bad" components

contours_graphic[good_ixs].colors = "green"
contours_graphic[bad_ixs].colors = "red"

# %%
# #### Remove the "bad" components

# sorting the "good" neurons based on preferred directions
sorted_ixs = tuning_curves.iloc[:,good_ixs].idxmax().sort_values().index.values
print(sorted_ixs)

# %%

# filter dataset based on sortex indices
temporal_data = temporal_data[:,sorted_ixs]
contours = [contours[i] for i in sorted_ixs]

# %%
# #### Plot only the "good" components

# only plot the good indices
nap_figure[0,0].remove_graphic(contours_graphic)
contours_graphic = nap_figure[0,0].add_line_collection(data=contours, colors="w")

# %%
# ## Make a plot of the calcium traces as a `LineStack`
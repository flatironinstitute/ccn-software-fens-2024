# -*- coding: utf-8 -*-

"""
# Neuroscience using `fastplotlib` and `pynapple`

"""

# %%
#
# This notebook will build up a complex visualization using `fastplotlib`, in conjunction with `pynapple`, to show how
# `fastplotlib` can be a powerful tool in analysis and visualization of neural data!

import warnings

warnings.simplefilter('ignore')

# %%
import fastplotlib as fpl
import pynapple as nap
from ipywidgets import FloatSlider, Layout, VBox
from sidecar import Sidecar
from workshop_utils import TimeStore
from IPython.display import display
import numpy as np
from skimage import measure
import workshop_utils

# %%

import warnings

warnings.simplefilter('ignore')

# %%
# ## Load the data
#
# Recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded is the postsubiculum
# - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in
# a specific direction.

data = nap.load_file(workshop_utils.fetch_data("A0634-210617.nwb"))

print(data)

# %%
# ## View the behavior and calcium data
#
# **NOTE:** We are going to be using a WIP TimeStore model to help synchronize our visualization in time.
# Hopefully, by the end of the summer we will have developed a tool
# ([`pynaviz`](https://github.com/pynapple-org/pynaviz)) that makes these visualizations and
# synchronizations even easier :D

time_store = TimeStore()

# %%
#

# behavior shape (time, x, y)
behavior_data = data["beh_video"]
print(behavior_data.shape)

# %%
#

# calcium shape (time, x, y)
calcium_data = data["calcium_video"]
print(calcium_data.shape)

# %%
#
# ### Minimize our view of the data to where both behavior and position data are available:

start_time = data["position_time_support"]["start"][0]
end_time = data["position_time_support"]["end"][0]
(start_time, end_time)

# %%
#
# ### Create a plot for the calcium and behavior video

nap_figure = fpl.Figure(shape=(1, 2), names=[["raw", "behavior"]])

calcium_graphic = nap_figure["raw"].add_image(data=calcium_data[0], name="raw_frame", cmap="gnuplot2")
behavior_graphic = nap_figure["behavior"].add_image(data=behavior_data[0], cmap="gray")

# %%
#
# ###  Create a slider that updates the behavior and calcium videos using `pyanapple`

synced_time = FloatSlider(min=start_time, max=end_time, step=0.01, description="s", layout=Layout(width="60%"))

# %%
# ### Need to subtract the min from every frame in order to prevent a vignette effect

min_frame = calcium_data.min(axis=0)

# %%
# ### Add the components of our visualization to the `TimeStore` model to be synchronized

# add the slider
time_store.subscribe(subscriber=synced_time)

# add our calcium data
time_store.subscribe(subscriber=calcium_graphic, data=calcium_data)


# add our behavior data
# pass function to subtract min
def substract_min(frame):
    global min_frame

    return frame - min_frame


time_store.subscribe(subscriber=behavior_graphic, data=behavior_data, data_filter=substract_min)

# %%
# ### View the plot

# we are going to use `sidecar` to organize our visualization better :D
sc = Sidecar()
with sc:
    display(VBox([nap_figure.show(), synced_time]))


# %%

# ## Calculate the spatial contours and overlay them on the raw calcium data

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

# add the calculated contours as an overlay to the calcium video
contours_graphic = nap_figure["raw"].add_line_collection(data=contours, colors="w")

# %%
#
# **It is very easy to see that many of the identified neurons may be "bad" candidates. Let's remove them
# from the dataset as we go on in our anaylsis.**

# %%
#
# ## Select only head-direction neurons

# get the temporal data (calcium transients) from the nwb notebook
temporal_data = data["RoiResponseSeries"][:]
temporal_data

# %%

# compute 1D tuning curved based on head angle
head_angle = data["ry"]

tuning_curves = nap.compute_1d_tuning_curves_continuous(temporal_data, head_angle, nb_bins=120)

# %%
# ### Select the top 50 components

# select good components
good_ixs = list(np.argsort(np.ptp(tuning_curves, axis=0))[-50:])
bad_ixs = list(np.argsort(np.ptp(tuning_curves, axis=0))[:-50])

# %%
# ### Color the "good" and "bad" components

contours_graphic[good_ixs].colors = "green"
contours_graphic[bad_ixs].colors = "white"

# %%
# #### Sort the "good" components based on preferred head direction

sorted_ixs = tuning_curves.iloc[:, good_ixs].idxmax().sort_values().index.values

# %%
# #### Filter the dataset to only use the sorted "good" components
#
# In the rest of the demo we will only be using the sub-sampled components.

temporal_data = temporal_data[:, sorted_ixs]
contours = [contours[i] for i in sorted_ixs]

# %%
# ### Plot only the "good" components

# remove the graphic of all the components
nap_figure[0, 0].remove_graphic(contours_graphic)

# re-plot only the good ixs
contours_graphic = nap_figure[0, 0].add_line_collection(data=contours, colors="w")

# %%
# ## Make a plot of the calcium traces as a `LineStack`

# create a figure
tstack_fig = fpl.Figure(shape=(2, 1))

# %%

# we need to transpose our temporal data so that it is (# components, time (s))
raw_temporal = temporal_data.to_numpy().T

# use 'hsv' colormap to represent preferred head direction
tstack_graphic = tstack_fig[0, 0].add_image(data=raw_temporal, cmap="plasma", name="temporal-stack")

# %%
# #### Add a `LinearSelector` that we can map to our behavior and calcium videos

tstack_selector = tstack_graphic.add_linear_selector()
vertical_selector = tstack_selector.add_linear_selector(axis="y")

# %%
# #### Subscribe our `LinearSelector` to the `TimeStore` model

time_store.subscribe(tstack_selector, temporal_data.rate)

# %%
# ### View the plot

sc = Sidecar()

with sc:
    display(VBox([nap_figure.show(), tstack_fig.show(maintain_aspect=False), synced_time]))

# %%
# ### Make an interactive plot to view the tuning curves for each component
#
# #### Initialize the conditions

# select the first component
ix = 0

# set the first component colors to magenta
contours_graphic[ix].colors = "green"

# get the tuning curve of the first component
tuning_ix = sorted_ixs[ix]

tuning_curve = tuning_curves.T.iloc[tuning_ix]

# add the tuning curve to the plot as a line
tuning_graphic = tstack_fig[1,0].add_line(data=tuning_curve, offset=(0,0,0))

tstack_fig[1,0].auto_scale(maintain_aspect=False)


# %%

# add an event handler to make the plot interactive
@vertical_selector.add_event_handler("selection")
def update_selected_trace(ev):
    ix = ev.get_selected_index()

    # reset the colors of the components to white
    contours_graphic.colors = "w"

    # set the selected component colors to magenta
    contours_graphic[ix].colors = "green"

    nap_figure["raw"].camera.show_object(contours_graphic[ix].world_object)

    # get tuning curve of the selected component
    tuning_ix = sorted_ixs[ix]

    tuning_curve = tuning_curves.T.iloc[tuning_ix]

    # remove the current tuning curve add the new one
    # global tuning_graphic
    tstack_fig[1, 0].graphics[0].data[:, 1] = tuning_curve

# %%

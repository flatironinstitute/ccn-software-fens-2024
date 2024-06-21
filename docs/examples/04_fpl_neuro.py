# -*- coding: utf-8 -*-

"""# Neuroscience 🧠 using `fastplotlib` 🦜 and `pynapple` 🍍

!!! warning

    mkdocs won't build this notebook, so the outputs won't be visible on the
    https://flatironinstitute.github.io/ website. To see the outputs, download this
    notebook and run it locally.

"""

# %%
#
# This notebook will build up a complex visualization using `fastplotlib`, in conjunction with `pynapple`, to show how
# `fastplotlib` can be a powerful tool in analysis and visualization of neural data!

# %%
import fastplotlib as fpl
import pynapple as nap
import numpy as np
from ipywidgets import Layout, VBox, FloatSlider
from skimage import measure
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sidecar import Sidecar
import workshop_utils
from IPython.display import display

# %%

import warnings

warnings.simplefilter('ignore')
fpl.config.party_parrot = True

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
# **NOTE:** We are going to be using a work-in-progress `TimeStore` model to help synchronize our visualization in time.
# Hopefully, by the end of the summer we will have developed a tool
# ([`pynaviz`](https://github.com/pynapple-org/pynaviz)) that makes these visualizations and
# synchronizations even easier :D

time_store = workshop_utils.TimeStore()

# %%
# Behavior data and shape 🐭

behavior_data = data["beh_video"]
print(behavior_data.shape)  # (time, x, y)

# %%
# Calcium data and the shape 🔬

calcium_data = data["calcium_video"]
print(calcium_data.shape)  # (time, x, y)

# %%
# The behavior tracks need to be scaled

print(data["x"].min(), data["x"].max())
print(data["z"].min(), data["z"].max())

# %%
# Scale the behavior tracks data with respect to the behavior dims movie

data["x"] = data["x"] + np.abs(data["x"].min())
data["x"] = data["x"] * behavior_data.shape[1]

data["z"] = data["z"] + np.abs(data["z"].min())
data["z"] = data["z"] * behavior_data.shape[2]

# %%
# Array of the behavior tracks

tracks_data = np.column_stack([data["z"], data["x"]])

print(tracks_data)

# %%
#
# #### Set our view of the data to where both behavior and position data are available:

behavior_data = behavior_data.restrict(data["position_time_support"])
calcium_data = calcium_data.restrict(data["position_time_support"])

print(data["position_time_support"].start[0], data["position_time_support"].end[0])

# %%

# calculate min frame across movie
# remove vignette effect from 1p endoscopic imaging
min_frame = calcium_data.min(axis=0)

# just to show you what this looks like
iw = fpl.ImageWidget(min_frame)
iw.show()

# %%

# close the plot
iw.close()

# %%
#
# ## Create a big viz for calcium and behavior video! 🎨

# make figure, calcium on left, behavior on right
nap_figure = fpl.Figure(shape=(1, 2), names=[["calcium", "behavior"]])

# image graphic to display current calcium frame
calcium_graphic = nap_figure["calcium"].add_image(data=calcium_data[0] - min_frame, name="calcium_frame",
                                                  cmap="gnuplot2")

# a UI tool to help set and visualize vmin-vmax
hlut = fpl.widgets.histogram_lut.HistogramLUT(data=calcium_data, image_graphic=calcium_graphic)
# add this to the right dock
nap_figure["calcium"].docks["right"].add_graphic(hlut)
nap_figure["calcium"].docks["right"].size = 80
nap_figure["calcium"].docks["right"].auto_scale(maintain_aspect=False)
nap_figure["calcium"].docks["right"].controller.enabled = False

# image graphic to display current behavior video frame
behavior_graphic = nap_figure["behavior"].add_image(data=behavior_data[0], cmap="gray")

# line to display the behavior tracks
tracks_graphic = nap_figure["behavior"].add_line(tracks_data, cmap="winter", thickness=.1, alpha=0.5, offset=(12, 0, 2))

# %%
#
# #### Create a slider that updates the behavior and calcium videos using `pynapple`

# This time will be in seconds
synced_time = FloatSlider(min=data["position_time_support"].start, max=data["position_time_support"].end, step=0.01,
                          description="s")


# auto-resize slider
@nap_figure.renderer.add_event_handler("resize")
def resize_slider(ev):
    synced_time.layout = Layout(width=f"{ev.width}px")


# %%
# #### Add the components of our visualization to the `TimeStore` model to be synchronized 🕰️

# add the slider
time_store.subscribe(subscriber=synced_time)


def substract_min(frame):
    """subtract min frame from current frame"""
    global min_frame

    return frame - min_frame


# add our calcium data
time_store.subscribe(subscriber=calcium_graphic, data=calcium_data, data_filter=substract_min)

# add our behavior data
time_store.subscribe(subscriber=behavior_graphic, data=behavior_data)

# %%
# **Here we are going to use `sidecar` to organize our visualization better :D**

sc = Sidecar()
with sc:
    display(VBox([nap_figure.show(), synced_time]))

# %%
# ### Visualize head angle just by setting the cmap transform, it's that simple! 🪄

# set cmap transform from "ry" head angle
tracks_graphic.cmap.transform = data["ry"].to_numpy()

# change to a heatmap more suitable for this data
tracks_graphic.cmap = "hsv"

# %%

# ### Visualize other kinematics just by setting the cmap transform! :D
def get_velocity(array):
    return np.gradient(np.abs(gaussian_filter1d(array, sigma=10)))


# velocity if the y-direction
tracks_graphic.cmap.transform = get_velocity(data["z"].to_numpy())
tracks_graphic.cmap = "seismic"  # diverging colormap, velocities are negative and positive

# %%

# velocity in the x-direction
tracks_graphic.cmap.transform = get_velocity(data["x"].to_numpy())

# %%
# Let's go back to head direction

tracks_graphic.cmap.transform = data["ry"].to_numpy()
tracks_graphic.cmap = "hsv"

# %%
# ## Visualize Calcium Imaging ROIs
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
#  #### Add the calculated contours as an overlay to the calcium video
contours_graphic = nap_figure["calcium"].add_line_collection(data=contours, colors="w")

# %%
#
# **It is very easy to see that many of the identified neurons may be "bad" candidates. Let's remove them
# from the dataset as we go on in our anaylsis.**

# %%
#
# ### Select only head-direction neurons

# get the temporal data (calcium transients) from the nwb notebook
temporal_data = data["RoiResponseSeries"][:].restrict(data["position_time_support"])
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

contours_graphic[good_ixs].colors = "w"
contours_graphic[bad_ixs].colors = "magenta"

# %%
# ### Sort the "good" components based on preferred head direction

sorted_ixs = tuning_curves.iloc[:, good_ixs].idxmax().sort_values().index.values

print(sorted_ixs)

# %%
# #### Filter the dataset to only use the sorted "good" components
#
# In the rest of the demo we will only be using the sub-sampled components.

temporal_data = temporal_data[:, sorted_ixs]
contours = [contours[i] for i in sorted_ixs]

# %%
# ### Plot only the "good" components

# remove the graphic of all the components
nap_figure["calcium"].remove_graphic(contours_graphic)

# re-plot only the good ixs
contours_graphic = nap_figure[0, 0].add_line_collection(data=contours, colors="w", alpha=0.8)

# %%
# ## Visualize all calcium tracing using an `ImageGraphic` to display a heatmap

# create a figure, 2 rows, 1 column
temporal_fig = fpl.Figure(shape=(2,1), names=[["temporal-heatmap"], ["tuning-curve"]])

# %%

# we need to transpose our temporal data so that it is (# components, time (s))
raw_temporal = temporal_data.to_numpy().T

# use 'hsv' colormap to represent preferred head direction
heatmap_graphic = temporal_fig[0,0].add_image(data=raw_temporal, cmap="plasma", name="traces")

# %%
# #### Add a `LinearSelector` that we can map to our behavior and calcium videos

time_selector = heatmap_graphic.add_linear_selector()

# add a selector for the y-axis to select components
component_selector = heatmap_graphic.add_linear_selector(axis="y")

# %%

# subscribe x-axis selector to timestore
time_store.subscribe(subscriber=time_selector, multiplier=temporal_data.rate)

# %%
# ### Let's view everything together

@nap_figure.renderer.add_event_handler("resize")
def resize_temporal_fig(ev):
    temporal_fig.canvas.set_logical_size(ev.width, 300)

sc = Sidecar()

with sc:
    display(VBox([nap_figure.show(), temporal_fig.show(maintain_aspect=False), synced_time]))

# %%

# select the first component
ix = 0

# set the first component colors to magenta
contours_graphic[ix].colors = "green"

# get the tuning curve of the first component
tuning_ix = sorted_ixs[ix]

tuning_curve = tuning_curves.T.iloc[tuning_ix]

# add the tuning curve to the plot as a line
tuning_graphic = temporal_fig["tuning-curve"].add_line(data=tuning_curve, offset=(0,0,0))
temporal_fig["tuning-curve"].auto_scale(maintain_aspect=False)


# %%
# ### Add an event handler that allows us to "scroll" through the components and tuning curves :D
@component_selector.add_event_handler("selection")
def update_selected_trace(ev):
    ix = ev.get_selected_index()

    # reset the colors of the components to white
    contours_graphic.colors = "w"

    # set the selected component colors to magenta
    contours_graphic[ix].colors = "green"

    nap_figure["calcium"].camera.show_object(contours_graphic[ix].world_object)

    # get tuning curve of the selected component
    tuning_ix = sorted_ixs[ix]

    tuning_curve = tuning_curves.T.iloc[tuning_ix]

    # remove the current tuning curve add the new one
    # global tuning_graphic
    temporal_fig["tuning-curve"].graphics[0].data[:, 1] = tuning_curve
    temporal_fig["tuning-curve"].auto_scale(maintain_aspect=False)

# %%
# ## Downstream analysis, view a PCA of the calcium

# ### Perform PCA

pca = PCA(n_components=3)

zscored = zscore(np.sqrt(temporal_data.to_numpy()), axis=1)
calcium_pca = pca.fit_transform(gaussian_filter1d(zscored, sigma=3))

# %%
# #### Plot the PCA results
#
# To get a proper colormap transform based on the head angle data, need to interpolate the timescale.

# restrict the head angle data
ry_restrict = data["ry"].restrict(data["position_time_support"])

# %%

x = np.arange(0, temporal_data.shape[0])
xp = np.linspace(0, temporal_data.shape[0], ry_restrict.shape[0])

# interpolate
ry_transform = np.interp(x, xp, fp=ry_restrict)  # use the y-values

# %%
# #### Make plot

fig_pca = fpl.Figure(
    cameras="3d",
    controller_types="orbit",
)
fig_pca[0, 0].add_scatter(calcium_pca, cmap="hsv", cmap_transform=ry_transform, sizes=4, alpha=0.4)
marker_graphic = fig_pca[0, 0].add_scatter(calcium_pca[0], sizes=10)

fig_pca.show()

# %%
# #### Subscribe the PCA marker to the `TimeStore` model

# create a `pynapple.TsdFrame` for the PCA data
pca_data = nap.TsdFrame(t=temporal_data.t, d=calcium_pca)

time_store.subscribe(subscriber=marker_graphic, data=pca_data, multiplier=temporal_data.rate)

# %%
# Can change the controller

fig_pca[0, 0].controller = "fly"

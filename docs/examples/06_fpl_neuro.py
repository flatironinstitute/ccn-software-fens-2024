# -*- coding: utf-8 -*-

"""
# Neuroscience using `fastplotlib` and `pynapple`

"""

# %%
#
# This notebook will build up a complex visualization using `fastplotlib`, in conjunction with `pynapple`, to show how
# `fastplotlib` can be a powerful tool in analysis and visualization of neural data!


import fastplotlib as fpl
import pynapple as nap
from ipywidgets import IntSlider, Layout, VBox
from sidecar import Sidecar
from utils import TimeStore
from IPython.display import display

# %%
# ## Load the data
#
# Recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded is the postsubiculum
# - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in
# a specific direction.

data = nap.load_file("./data.nwb")

print(data)

# %%
# ## View the behavior and calcium data
#
# NOTE: We are going to be using a WIP TimeStore model to help synchronize our visualization in time.
# Hopefully, by the end of the summer we will have developed a tool
# ([`pynaviz`](https://github.com/pynapple-org/pynaviz)) that makes these visualizations and
# synchronizations even easier :D

time_store = TimeStore()

# %%
#

# behavior shape
behavior_data = data["beh_video"]
print(behavior_data.shape)

# %%
#

# calcium shape
calcium_data = data["calcium_video"]
print(calcium_data.shape)

# %%
#
# ### Minimize our view of the data to where both behavior and position data are available:

frame_min = data["position_time_support"]["start"][0]
frame_max = data["position_time_support"]["end"][0]
print((frame_min, frame_max))

# %%
#
# ### Create a plot for the calcium and behavior video

nap_figure = fpl.Figure(shape=(1,2), names=[["raw", "behavior"]])

calcium_graphic = nap_figure["raw"].add_image(data=calcium_data[0], name="raw_frame", cmap="viridis")
behavior_graphic = nap_figure["behavior"].add_image(data=behavior_data[0], cmap="gray")

# %%
#
# ###  Create a slider that updates the behavior and calcium videos using `pyanapple`

synced_time = IntSlider(min=frame_min, max=frame_max, step=1, description="s", layout=Layout(width="60%"))

# %%
# ### Add the components of our visualization to the TimeStore model to be synchronized

# add the slider
time_store.subscribe(subscriber=synced_time)

# add our calcium data
time_store.subscribe(subscriber=calcium_graphic, data=calcium_data)

# add out behavior data
time_store.subscribe(subscriber=behavior_graphic, data=behavior_data)

# %%
# ### View the plot

# we are going to use `sidecar` to organize our visualization better :D
sc = Sidecar()
with sc:
    display(VBox([nap_figure.show(), synced_time]))

# %%

# manually set the vmin/vmax of the calcium data
nap_figure["raw"]["raw_frame"].vmax = 205
nap_figure["raw"]["raw_frame"].vmin = 25

# %%



# -*- coding: utf-8 -*-

"""
# Introduction to `fastplotlib`

## `fastplotlib` API

### 1. Graphics - objects that are drawn
- `Image`, `Line`, `Scatter`
- Collections - `LineCollection`, `LineStack` (ex: neural timeseries data)

    #### a) Graphic Properties
  - Common: `Name`, `Offset`, `Rotation`, `Visible`, `Deleted`
  - Graphic Specific: `ImageVmin`, `ImageVmax`, `VertexColors`, etc.

    #### b) Selectors
  - `LinearSelector` - horizontal or vertical line slider
  - `LinearRegionSelector` - horizontal or vertical resizable region selection

### 2. Layouts
- `Figure` - a single plot or a grid of subplots

### 3. Widgets - high level widgets to make repetitive UIs easier
- `ImageWidget`- widget for `ImageGraphic` data with dims: `xy`, `txy`, `tzxy`
- Sliders, support window functions, etc.

"""

# %%
# This notebook will go through some basic components of the `fastplotlib` API including how to instantiate a plot,
# add graphics to a plot, and subsequently interact with the plot.

# %%
# **The example images are from `imageio` so you will need to install it for this example notebook. But `imageio`
# is not required to use fasptlotlib**

# uncomment and install if needed
#! pip install imageio

# %%
#

import imageio.v3 as iio

# %%
# By default, `fastplotlib` will enumerate the available devices and highlight which device has been selected by
# default when importing the module.

import fastplotlib as fpl
import numpy as np
from itertools import product

# %%
# ## Simple Image

# create a `Figure` instance
# by default the figure will have 1 subplot
fig = fpl.Figure(size=(600, 500))

# get a grayscale image
data = iio.imread("imageio:camera.png")

# plot the image data
image_graphic = fig[0, 0].add_image(data=data, name="sample-image")

# show the plot
fig.show(sidecar=True)

# %%
# **Use the handle on the bottom right corner of the canvas to resize it.
# You can also pan and zoom using your mouse!**
#
# Changing graphic **properties**

image_graphic.cmap = "viridis"

# %%
# ### Slicing data
#
# **Most properties, such as data, support slicing!**
#
# Our image data is of shape [n_rows, n_columns]

print(image_graphic.data.value.shape)

image_graphic.data[::15, :] = 1
image_graphic.data[:, ::15] = 1

# %%
# **Fancy indexing**

image_graphic.data[data > 175] = 255

# %%
# **Adjust vmin vmax**

image_graphic.vmin = 50
image_graphic.vmax = 150

# %%
# **Reset the vmin vmax**

image_graphic.reset_vmin_vmax()

# %%
# ### Change graphic properties {.strip-code}{.keep-text}
#
# Now that we have a better idea of how graphic properties can be dynamically changed, let's practice :D
#
# **Question:** Can you change the data of the image to be random data of the same shape? Hint: Use `np.random.rand()`
# to create the new data.

# create new random dat of the same shape as the original image
new_data = np.random.rand(*image_graphic.data.value.shape)

# update the data
image_graphic.data = new_data

# %%
# **Question:** Can you change the colormap of the data? See [here](https://matplotlib.org/stable/gallery/color/colormap_reference.html) for a list of colormaps.

image_graphic.cmap = "hsv"

# %% {.keep-code}
#

# close the plot
fig.close()

# %%
# ### Image updates {.keep-code}{.keep-text}
# This examples show how you can define animation functions that run on every render cycle.

# create another `Figure` instance
fig_v = fpl.Figure(size=(600, 500))

fig.canvas.max_buffered_frames = 1

# make some random data again
data = np.random.rand(512, 512)

# plot the data
fig_v[0,0].add_image(data=data, name="random-image")

# a function to update the image_graphic
# a plot will pass its plot instance to the animation function as an arugment

def update_data(plot_instance):
    new_data = np.random.rand(512, 512)
    plot_instance[0,0]["random-image"].data = new_data

#add this as an animation function
fig_v.add_animations(update_data)

fig_v.show(sidecar=True)

# %%
# ### Adding another animation function {.strip-code}{.keep-text}
#
# **Question:** Can you add an animation function that toggles the colormap of the image between "plasma" and "hsv"?

# define a function to toggle the colormap
def update_cmap(plot_instance):
    if plot_instance[0,0]["random-image"].cmap == "plasma":
        plot_instance[0,0]["random-image"].cmap = "hsv"
    else:
        plot_instance[0,0]["random-image"].cmap = "plasma"

# add the animation function
fig_v.add_animations(update_cmap)

# %%
# Close the plot {.keep-code}
fig_v.close()

# %%
# ## 2D Line Plots

# This example plots a sine wave, cosine wave, and ricker wavelet and demonstrates how **Graphic Properties** can be
# modified by slicing!

# %%
# ### First generate some data

# linspace, create 100 evenly spaced x values from -10 to 10
xs = np.linspace(-10, 10, 100)
# sine wave
ys = np.sin(xs)
sine = np.dstack([xs, ys])[0]

# cosine wave
ys = np.cos(xs) + 5
cosine = np.dstack([xs, ys])[0]

# sinc function
a = 0.5
ys = np.sinc(xs) * 3 + 8
sinc = np.dstack([xs, ys])[0]

# %%
# **We will plot all of it on the same plot. Each line plot will be an individual `Graphic`, you can have any
# combination of graphics in a plot.**

# Create a figure
fig_lines = fpl.Figure()
# we will add all the lines to the same subplot
subplot = fig_lines[0, 0]

# plot sine wave, use a single color
sine_graphic = subplot.add_line(data=sine, thickness=5, colors="magenta")

# you can also use colormaps for lines!
cosine_graphic = subplot.add_line(data=cosine, thickness=12, cmap="autumn")

# or a list of colors for each datapoint
colors = ["r"] * 25 + ["purple"] * 25 + ["y"] * 25 + ["b"] * 25
sinc_graphic = subplot.add_line(data=sinc, thickness=5, colors = colors)

# show the plot
fig_lines.show(sidecar=True, sidecar_kwargs={"title": "lines"})

# %%
# ### Graphic properties support slicing! :D

# indexing of colors
cosine_graphic.colors[:15] = "magenta"
cosine_graphic.colors[90:] = "red"
cosine_graphic.colors[60] = "w"

# %%

# indexing to assign colormaps to entire lines or segments
sinc_graphic.cmap[10:50] = "gray"
sine_graphic.cmap = "seismic"

# %%

# more complex indexing, set the blue value directly from an array
cosine_graphic.colors[65:90, 0] = np.linspace(0, 1, 90-65)

# %%

# slicing data
bool_key = [True, True, True, False, False] * 20
sinc_graphic.data[bool_key, 1] = 7  # y vals to 1

# %%
# #### 2D Lines Practice {.strip-code}{.keep-text}
#
# **Question:** Can you change the colormap of the sine graphic to "hsv"?

sine_graphic.cmap = "hsv"

# %%
# **Question:** Can you change the color of first 50 data points of the sinc graphic to green?

sinc_graphic.colors[:50] = "g"

# %%
# **Question:** Can you to change the last 50 data points of the cosine graphic to equal the last 50 points of the
# sine graphic?

cosine_graphic.data[50:] = sine_graphic.data[50:]

# %%
# #### Capture changes to graphic properties as events
#
# **Two ways to add events in `fastplotlib`:**
#
#1) Using `graphic.add_event_handler(callback_func, "property")`
#
#2) Using a decorator:
#```
#@graphic.add_event_handler("property")
#def callback_func(ev):
#    pass
#```

# %%

# will print event data when the color of the cosine graphic changes
@cosine_graphic.add_event_handler("colors")
def callback_func(ev):
    print(ev.info)

# %%
# when the cosine graphic colors change, will also update the sine_graphic colors
def change_colors(ev):
   sine_graphic.colors[ev.info["key"]] = "w"

cosine_graphic.add_event_handler(change_colors, "colors")

# %%
cosine_graphic.colors[:10] = "g"

# %%
fig_lines.close()

# %%
# #### More 2D Lines Practice {.strip-code}{.keep-text}
#
# **Question:** Can you add an event handler (using either method) to update ?




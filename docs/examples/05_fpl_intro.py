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
# By default, `fastplotlib` will enumerate the available devices and highlight which device has been selected by
# default when importing the module.

import fastplotlib as fpl
import numpy as np
import imageio.v3 as iio

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

# %%
#

image_graphic.data[::15, :] = 1
image_graphic.data[:, ::15] = 1

# %%
# **Fancy indexing**

image_graphic.data[data > 175] = 255

# %%
# **Adjust vmin vmax**
# NOTE: vmin and vmax values range from (0, 255)

image_graphic.vmin = 50
image_graphic.vmax = 150

# %%
# **Reset the vmin vmax**

image_graphic.reset_vmin_vmax()

# %%
# ### Change graphic properties {.strip-code,.keep-text}
#
# Now that we have a better idea of how graphic properties can be dynamically changed, let's practice :D
#
# **Question:** Can you change the data of the image to be random data of the same shape? Hint: Use `np.random.rand()`
# to create the new data.
#
# Note: At first you should see a solid image. You will also need to rest the vmin/vmax.

# create new random data of the same shape as the original image
new_data = np.random.rand(*image_graphic.data.value.shape)

# update the data
image_graphic.data = new_data

# %%
# **Question:** Can you change the colormap of the image? See [here](https://matplotlib.org/stable/gallery/color/colormap_reference.html) for a list of colormaps.

image_graphic.cmap = "hsv"

# %% {.keep-code}
#

# close the plot
fig.close()

# %%
# ### Image updates {.keep-code,.keep-text}
# This examples show how you can define animation functions that run on every render cycle.

# create another `Figure` instance
fig_v = fpl.Figure(size=(600, 500))

fig.canvas.max_buffered_frames = 1

# make some random data again
data = np.random.rand(512, 512)

# plot the data
fig_v[0, 0].add_image(data=data, name="random-image")


# a function to update the image_graphic
# a plot will pass its plot instance to the animation function as an argument

def update_data(plot_instance):
    new_data = np.random.rand(512, 512)
    plot_instance[0, 0]["random-image"].data = new_data


#add this as an animation function
fig_v.add_animations(update_data)

fig_v.show(sidecar=True)

# %%
#

# close the plot
fig_v.close()

# %%
# ### Image Practice {.strip-code,.keep-text}
#
# **Question:** Can you do the following:
#
# - create a new plot called `practice_fig`
#
# - add an `ImageGraphic` with the following characteristics to the figure:
#
#      - random data in the shape (512, 512)
#
#      - colormap "viridis"
#
#      - name "random-image"
#
# - set the top-left and bottom-right quadrants of the data to 1 using slicing :D
#
# - add an animation function that updates the top-right and bottom-left quadrants with new random data


practice_fig = fpl.Figure()

data = np.random.rand(512, 512)

practice_fig[0, 0].add_image(data=data, name="random-image", cmap="viridis")

# set the top-left and bottom-right quadrants of the data to 1
practice_fig[0, 0]["random-image"].data[:256, :256] = 1  # top-left
practice_fig[0, 0]["random-image"].data[256:, 256:] = 1  # bottom-right


# define an animation function to toggle the data
def update_data(plot_instance):
    # set the top-right and bottom-left quadrants with new random data
    plot_instance[0, 0]["random-image"].data[:256, 256:] = np.random.rand(256, 256)  # bottom-left
    plot_instance[0, 0]["random-image"].data[256:, :256] = np.random.rand(256, 256)  # top-right


# add the animation function
practice_fig.add_animations(update_data)

practice_fig.show()

# %%

# close the plot
practice_fig.close()

# %%
# ## 2D Line Plots
#
# This example plots a sine wave, cosine wave, and ricker wavelet and demonstrates how **Graphic Properties** can be
# modified by slicing!

# %%
# ### First generate some data

# linspace, create 100 evenly spaced x values from -10 to 10
xs = np.linspace(-10, 10, 100)
# sine wave
ys = np.sin(xs)
sine = np.column_stack([xs, ys])

# cosine wave
ys = np.cos(xs)
cosine = np.column_stack([xs, ys])

# sinc function
a = 0.5
ys = np.sinc(xs) * 3
sinc = np.column_stack([xs, ys])

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
cosine_graphic = subplot.add_line(data=cosine, thickness=12, cmap="autumn", offset=(0, 5, 0))

# or a list of colors for each datapoint
colors = ["r"] * 25 + ["purple"] * 25 + ["y"] * 25 + ["b"] * 25
sinc_graphic = subplot.add_line(data=sinc, thickness=5, colors=colors, offset=(0, 8, 0))

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


# %%

# use a qualitative cmap (https://matplotlib.org/stable/_images/sphx_glr_colormaps_006.png)
sine_graphic.cmap = "plasma"
# set the cmap transform
sine_graphic.cmap.transform = [4] * 25 + [1] * 25 + [3] * 25 + [8] * 25


# %%
#

# close the plot
fig_lines.close()

# %%
# We are going to use the same sine and cosine data from before and do some practice!

# Create a figure
fig_lines = fpl.Figure()

# we will add all the lines to the same subplot
subplot = fig_lines[0, 0]

# plot sine wave, use a single color
sine_graphic = subplot.add_line(data=sine, thickness=5, colors="magenta")

# you can also use colormaps for lines!
cosine_graphic = subplot.add_line(data=cosine, thickness=12, cmap="autumn", offset=(0, 5, 0))

# show the plot
fig_lines.show(sidecar=True, sidecar_kwargs={"title": "lines"})

# %%
# #### 2D Lines Practice {.strip-code,.keep-text}
# **Question:** Can you change the colormap of the sine_graphic to "hsv"?

sine_graphic.cmap = "hsv"

# %%
# **Question:** Can you change the color of first 50 data points of the sinc_graphic to green?

sinc_graphic.colors[:50] = "g"

# %%
# **Question:** Can you to change the color of last 50 data points of the cosine_graphic to equal the colors of the
# last 50 data points of the sine_graphic?

cosine_graphic.colors[50:] = sine_graphic.colors[50:]


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
# Examples:

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
# ### More Events :D
#

# generate some circles
def make_circle(center, radius: float, n_points: int = 75) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_points)
    xs = radius * np.sin(theta)
    ys = radius * np.cos(theta)

    return np.column_stack([xs, ys]) + center


# this makes 5 circles, so we can create 5 cmap values, so it will use these values to set the
# color of the line based by using the cmap as a LUT with the corresponding cmap_value
circles = list()
for x in range(0, 50, 10):
    circles.append(make_circle(center=(x, 0), radius=4, n_points=100))


# %%

fig = fpl.Figure()

circles_graphic = fig[0, 0].add_line_collection(data=circles, cmap="tab10", thickness=10)

# %%

fig.show()


# %%

# get graphic that is clicked and change the color
def click_event(ev):
    # reset the colors
    circles_graphic.cmap = "tab10"

    # change clicked circle color
    ev.graphic.colors = "w"


# %%

# for each circle in the collection, add click event handler
circles_graphic.add_event_handler(click_event, "click")


# %%
# #### Events Practice {.keep-text,.strip-code}
#
# **Question:** Can you add another event handler (using either method) to the circles_graphic that will change the
# thickness of a circle to 5 when you hover over it?
#
# Hint: the event type should be "pointer_move" :)

@circles_graphic.add_event_handler("pointer_move")
def hover_event(ev):
    # reset the thickness of all the circles
    circles_graphic.thickness = [10] * len(circles_graphic)

    # change the thickness of the circle that was hovered over
    ev.graphic.thickness = 5


# %%

# close the plot
fig.close()

# %%
# ## Selectors
#
# ### `LinearSelector`

fig = fpl.Figure()

# same sine data from before
sine_graphic = fig[0, 0].add_line(data=sine, colors="w")

# add a linear selector the sine wave
selector = sine_graphic.add_linear_selector()

fig[0, 0].auto_scale()

fig.show(maintain_aspect=False)


# %%

# change the color of the sine wave based on the location of the linear selector
@selector.add_event_handler("selection")
def set_color_at_index(ev):
    # get the selected index
    ix = ev.get_selected_index()
    # get the sine graphic
    g = ev.graphic.parent
    # change the color of the sine graphic at the index of the selector
    g.colors[ix] = "green"


# %%

# close the plot
fig.close()

# %%
# ### `LinearRegionSelector`

fig = fpl.Figure((2, 1))

# data to plot
xs = np.linspace(0, 10 * np.pi, 1_000)
sine = np.sin(xs)
sine += 100

# make sine along x axis
sine_graphic_x = fig[0, 0].add_line(np.column_stack([xs, sine]), offset=(10, 0, 0))

# add a linear region selector
ls_x = sine_graphic_x.add_linear_region_selector()  # default axis is "x"

# get the initial selected date of the linear region selector
zoomed_init = ls_x.get_selected_data()

# make a line graphic for displaying zoomed data
zoomed_x = fig[1, 0].add_line(zoomed_init)


@ls_x.add_event_handler("selection")
def set_zoom_x(ev):
    """sets zoomed x selector data"""
    # get the selected data
    selected_data = ev.get_selected_data()

    # remove the current zoomed data
    # and update with new selected data
    global zoomed_x
    fig[1, 0].remove_graphic(zoomed_x)
    zoomed_x = fig[1, 0].add_line(selected_data)
    fig[1, 0].auto_scale()


fig.show(maintain_aspect=False)

# %%

# close the plot
fig.close()

# %%
# ## For more examples, please see our gallery:
#
# ### https://fastplotlib.readthedocs.io/en/latest/_gallery/index.html

# %%
# ## For a more comprehensive intro to the library, please see our guide:

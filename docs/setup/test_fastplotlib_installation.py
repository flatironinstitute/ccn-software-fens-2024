# -*- coding: utf-8 -*-
""" # Test `fastplotlib` install

This notebook is intended to test your installation of `fastplotlib` and the associated libraries. If everything in this notebook runs properly, with no errors, then you're good to go!

Otherwise, come to the setup help session on Saturday afternoon.
"""

import fastplotlib as fpl
import numpy as np

# %%
# ## Test `Figure` and adding different types of `Graphics`

fig = fpl.Figure(shape=(2,2))

# %%
# create some random image and line data 
img = np.random.rand(512, 512)

# linspace, create 100 evenly spaced x values from -10 to 10
xs = np.linspace(-10, 10, 100)
# sine wave
ys = np.sin(xs)
sine = np.column_stack([xs, ys])

# cosine wave
ys = np.cos(xs) + 5
cosine = np.column_stack([xs, ys])

# sinc function
a = 0.5
ys = np.sinc(xs) * 3 + 8
sinc = np.column_stack([xs, ys])

# random scatter plot data
n_points = 1_000

# dimensions always have to be [n_points, xyz]
dims = (n_points, 3)

clouds_offset = 15

# create some random clouds
normal = np.random.normal(size=dims, scale=5)
# stack the data into a single array
cloud = np.vstack(
    [
        normal - clouds_offset,
        normal,
        normal + clouds_offset,
    ]
)

# color each of them separately
colors = ["yellow"] * n_points + ["cyan"] * n_points + ["magenta"] * n_points

fig[0,0].add_image(data=img, name="random-img")
fig[0,1].add_line_collection(data=[sine, cosine, sinc], colors=["r", "g", "b"])
fig[1,0].add_scatter(data=cloud, colors=colors)

fig.show()

# %%
# ### Test `ImageWidget`

iw_data = np.random.rand(30, 512, 512)
iw = fpl.ImageWidget(data=iw_data, cmap="viridis")
iw.show()

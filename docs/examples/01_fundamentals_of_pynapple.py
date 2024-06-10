# -*- coding: utf-8 -*-
""" # Learning the fundamentals of pynapple

## Learning objectives {.keep-text}

- Instantiate the pynapple objects
- Make the pynapple objects interact
- Learn the core functions of pynapple


Let's start by importing the pynapple package and matplotlib to see if everything is correctly installed. 
If an import fails, you can do `!pip install pynapple matplotlib` in a cell to fix it.
"""

import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np

# %%
# For this notebook we will work with fake data. The following cells generate a set of variables that we will use to create the different pynapple objects.

var1 = np.random.randn(100) # Variable 1
tsp1 = np.arange(100) # The timesteps of variable 1

var2 = np.random.randn(10, 3) # Variable 2
tsp2 = np.arange(0, 100, 10) # The timesteps of variable 2
col2 = ['potato', 'banana', 'tomato'] # The name of each columns of var2

var3 = np.random.randn(1000, 4, 5) # Variable 3
tsp3 = np.arange(0, 100, 0.1) # The timesteps of variable 3

random_times_1 = np.array([3.14, 37.0, 42.0])
random_times_2 = np.array([10.0, 30, 50, 70, 90])
random_times_3 = np.sort(np.random.uniform(0, 100, 1000))

starts_1 = np.array([10, 60, 90])
ends_1 = np.array([20, 80, 95])

# %%
# This is a lot of variables to carry around. pynapple can help reduce the size of the workspace. Here we will instantiate all the different pynapple objects with the variables created above.
#
# Let's start with the simple ones. 
#
# **Question:** Can you instantiate the right pynapple objects for `var1`, `var2` and `var3`? Objects should be named respectively `tsd1`, `tsd2` and `tsd3`.

tsd1 = nap.Tsd(t=tsp1, d=var1)
tsd2 = nap.TsdFrame(t=tsp2, d=var2)
tsd3 = nap.TsdTensor(t=tsp3, d=var3)

# %%
# **Question:** Can you print them one by one?

print(tsd1)

# %%
# """# Introduction to time series objects

# The cell below creates artificial data sampled every 1 second. In this experiment, we start by collecting one data stream. Here *t* and *d* are the timestamps and data, respectively.
# """

# import numpy as np

# rng = np.random.default_rng(seed=42) #the answer to the ultimate question of life, the universe, and everything

# t = np.arange(0, 1000)

# d = np.random.randn(1000, 5, 5)
# d[:,0,0] += np.cos(t*0.05)

# """

# ---

# **Question 1** : Can you convert the data to the right pynapple object?"""

# imqging = nap.TsdTensor(t=t, d=d)

# """**Question 2:** Can you print your object to see its dimensions?"""

# print(imqging)

# """**Question 3**: If your datastream is a movie, can you extract the time series of the pixel at coordinate (0, 0)?"""

# imqging[:,0,0]

# """**Question 4**: We are interested in only the first 100 seconds and last 100 seconds of the recording. Can you restrict the first data stream to those epochs only? Don't forget to visualize the output of your operation."""

# ep = nap.IntervalSet(start=[0, 900], end=[100, 1000])
# print(ep)

# imaging_res = imqging.restrict(ep)
# print(imaging_res)

# print(imqging.time_support)
# print(imaging_res.time_support)

# """**Question 5:** Can you average all the pixels only from those two epochs?"""

# np.mean(imaging_res, (1,2))

# """**Question 6**: Can you average along the time axis during the same two epochs?"""

# np.mean(imaging_res, 0)

# """---

# # Using Ts and TsGroup

# Let's say you recorded neurons from thalamus and prefrontal cortex together.
# """

# t_thl_0 = nap.Ts(t=np.sort(np.random.uniform(0, 100, 1000)))
# t_thl_1 = nap.Ts(t=np.sort(np.random.uniform(0, 100, 2000)))
# t_pfc_0 = nap.Ts(t=np.sort(np.random.uniform(0, 100, 3000)))

# """**Question 7**: Can you group them together in a TsGroup and add the location to each?"""

# spktrain = nap.TsGroup({0: t_thl_0, 1: t_thl_1 , 2: t_pfc_0}, location = np.array(['thl', 'thl', 'pfc']))
# print(spktrain)

# """**Question 9**: Can you select only the neurons from the thalamus?"""

# spktrain_thl = spktrain.getby_category('location')['thl']
# print(spktrain_thl)

# """**Question 10** : Can you bin the spike trains of the thalamus in bins of 1 second?"""

# binSpk = spktrain_thl.count(1)
# print(binSpk)

# """**Question 10**: Can you select the epochs for which the multi unit spike count of thalamic neurons is above 30 spikes?"""

# mua = np.sum(binSpk,1)
# print(mua)

# epHgMua = mua.threshold(30).time_support
# print(epHgMua)

# """**Question 11**: Can you restrict the spikes of the prefrontal neurons to the epochs previously defined?"""

# spkPfcRes = spktrain.getby_category('location')['pfc'].restrict(epHgMua)
# print(spkPfcRes)

# """# Introduction to core functions

# In this next experiment, we are recording a feature sampled at 100 Hz and one neuron.
# """


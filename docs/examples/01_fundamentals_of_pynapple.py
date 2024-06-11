# -*- coding: utf-8 -*-
""" # Learning the fundamentals of pynapple

## Learning objectives {.keep-text}

- Instantiate the pynapple objects
- Make the pynapple objects interact
- Use numpy with pynapple
- Slicing pynapple objects
- Learn the core functions of pynapple


Let's start by importing the pynapple package and matplotlib to see if everything is correctly installed. 
If an import fails, you can do `!pip install pynapple matplotlib` in a cell to fix it.
"""
# %%
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
random_times_2 = np.array([10.0, 30, 50, 70])
random_times_3 = np.sort(np.random.uniform(10, 80, 100))

starts_1 = np.array([10000, 60000, 90000]) # starts of an epoch in `ms`
ends_1 = np.array([20000, 80000, 95000]) # ends in `ms`

# %%
# ## Instantiate pynapple objects {.keep-text}
#
# This is a lot of variables to carry around. pynapple can help reduce the size of the workspace. Here we will instantiate all the different pynapple objects with the variables created above.
#
# Let's start with the simple ones. 
#
# **Question:** Can you instantiate the right pynapple objects for `var1`, `var2` and `var3`? Objects should be named respectively `tsd1`, `tsd2` and `tsd3`. Don't forget the column name for `var2`.

tsd1 = nap.Tsd(t=tsp1, d=var1)
tsd2 = nap.TsdFrame(t=tsp2, d=var2, columns = col2)
tsd3 = nap.TsdTensor(t=tsp3, d=var3)

# %%
# **Question:** Can you print `tsd1`?

print(tsd1)

# %%
# **Question:** Can you print `tsd2`?
print(tsd2)

# %%
# **Question:** Can you print `tsd3`?
print(tsd3)

# %%
# **Question:** Can you create an `IntervalSet` called `ep` out of `starts_1` and `ends_1` and print it? Be careful, times given above are in `ms`.

ep = nap.IntervalSet(start=starts_1, end=ends_1, time_units='ms')
print(ep)

# %%
# The experiment generated a set of timestamps from 3 different channels.
#  
# **Question:** Can you instantiate the corresponding pynapple object (`ts1`, `ts2`, `ts3`) for each one of them?

ts1 = nap.Ts(t=random_times_1)
ts2 = nap.Ts(t=random_times_2)
ts3 = nap.Ts(t=random_times_3)

# %%
# This is a lot of timestamps to carry around as well.
#
# **Question:** Can you instantiate the right pynapple object (call it `tsgroup`) to group them together?

tsgroup = nap.TsGroup({0:ts1, 1:ts2, 2:ts3})

# %%
# **Question:** ... and print it?

print(tsgroup)

# %%
# ## Interaction between pynapple objects {.keep-text}
#
# We reduced 12 variables in our workspace to 5 using pynapple. Now we can see how the objects interact.
#
# **Question:** Can you print the `time_support` of `TsGroup`?

print(tsgroup.time_support)

# %%
# The experiment ran from 0 to 100 seconds and as you can see, the `TsGroup` object shows the rate. But the rate is not accurate as it was computed over the default `time_support`.
#
# **Question:** can you recreate the `tsgroup` object passing the right `time_support` during initialisation?

tsgroup = nap.TsGroup({0:ts1, 1:tsd2, 2:ts3}, time_support = nap.IntervalSet(0, 100))

# %%
# **Question:** Can ou print the `time_support` and `rate` to see how they changed?

print(tsgroup.time_support)
print(tsgroup.rate)

# %%
# Now you realized the variable `tsd1` has some noise. The good signal is between 10 and 30 seconds and  50 and 100. 
# 
# **Question:** Can you create an `IntervalSet` object called `ep` and use it to restrict the variable `tsd1`?

ep = nap.IntervalSet(start=[10, 50], end=[30, 100])

tsd1 = tsd1.restrict(ep)

# %%
# You can print `tsd1` to check that the timestamps are in fact within `ep`.
# You can also check the `time_support` of `tsd1` to see that it has been updated.
#
print(tsd1)
print(tsd1.time_support)

# %%
# ## Numpy & pynapple
#
# Pynapple objects behaves very similarly like numpy array. They can be sliced withe the following syntax :
# 
#   `tsd[0:10] # First 10 elements`
#
# Arithmetical operations are available as well :
# 
#   `tsd = tsd + 1`
#
# Finally numpy functions works directly. Let's imagine `tsd3` is a movie with frame size (4,5).
# **Question:** Can you compute the average frame along the time axis using `np.mean` and print the result?

print(np.mean(tsd3, 0))

# %%
# **Question:**: can you compute the average of `tsd2` along the column axis and print it?

print(np.mean(tsd2, 1))

# %%
# Notice how the output in the second case is still a pynapple object.
# In most cases, applying a numpy function will return a pynapple object if the time index is preserved.
#
# ## Slicing pynapple objects {.keep-text}

#Inteval Set
# TsGroup

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


# %%

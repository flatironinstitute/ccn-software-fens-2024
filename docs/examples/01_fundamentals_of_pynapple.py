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

var2 = np.random.randn(100, 3) # Variable 2
tsp2 = np.arange(0, 100, 1) # The timesteps of variable 20
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
# **Question:** Can you create an `IntervalSet` object called `ep_signal` and use it to restrict the variable `tsd1`?

ep_signal = nap.IntervalSet(start=[10, 50], end=[30, 100])

tsd1 = tsd1.restrict(ep_signal)

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
#
# Multiple methods exists to slice pynapple object. This parts reviews them.
#
# `IntervalSet` also behaves like numpy array.
# **Question:** Can you extract the first and last epoch of `ep` in a new `IntervalSet`?

print(ep[[0,2]])

# %%
# Sometimes you want to get a data point as close as possible in time to another timestamps.
# **Question:** Using the `get` method, can you get the data point from `tsd3` as close as possible to the time 50.1 seconds?

print(tsd3.get(50.1))

# %%
# ## `TsGroup` manipulation
#
# `TsGroup` is under the hood a python dictionnary but the capabilities have been extented.
# **Question:** Can you run the following command `tsgroup['planet'] = ['mars', 'venus', 'saturn']`

tsgroup['planet'] = ['mars', 'venus', 'saturn']

# %%
# **Question:** ... and print it?

print(tsgroup)

# %%
# After initialization, metainformation can only be added. Running the following command will raise an error: `tsgroup[3] = np.random.randn(3)`.
#
# From there, you can slice using the Index column (i.e. `tsgroup[0]`->nap.Ts, `tsgroup[[0,2]]` -> nap.TsGroup). 
#
# But more interestingly you can also slice using the metadata. There are multiple methods for it : `getby_category`, `getby_threshold`, `getby_intervals`.
#
# **Question:** Can you select only the elements of `tsgroup` with rate below 1Hz?

tsgroup.getby_threshold("rate", 1, "<")

tsgroup[tsgroup.rate < 1.0]


# %%
# ## Core functions of pynapple
#
# This part focuses on the most important core functions of pynapple. 
#
# **Question:** Using the `count` function, can you count the number of events within 1 second bins for `tsgroup` over the `ep_signal` intervals?

count = tsgroup.count(1, ep_signal)

# %%
# Pynapple works directly with matplotlib. Passing a time series object to `plt.plot` will display the figure with the correct time axis.
#
# **Question:** In two subplots, can you show the count and events over time?

plt.figure()
ax = plt.subplot(211)
plt.plot(count, 'o-')
plt.subplot(212, sharex=ax)
plt.plot(tsgroup.restrict(ep_signal).to_tsd(), 'o')
plt.show()

# %%
# From a set of timestamps, you want to assign them a set of values with the closest point in time of another time series.
#
# **Question:** Using the function `value_from`, can you assign values to `ts2` from the `tsd1` time series and call the output `new_tsd`?

new_tsd = ts2.value_from(tsd1)

# %%
# **Question:** Can you plot together `tsd1`, `ts2` and `new_tsd`?

plt.figure()
plt.plot(tsd1)
plt.plot(new_tsd, 'o-')
plt.plot(ts2.fillna(0), 'o')
plt.show()

# %%
# **Question:** 
# One important aspect of data analysis is to bring data to the same size. Pynapple provides the `bin_average` function to downsample data. 
# 
# **Question:** Can you downsample `tsd2` to one time point every 5 seconds?

new_tsd2 = tsd2.bin_average(5.0)

# %%
# **Question:** Can you plot the `tomato` column from `tsd2` as well as the downsampled version?

plt.figure()
plt.plot(tsd2['tomato'])
plt.plot(new_tsd2['tomato'], 'o-')
plt.show()

# %%
# For `tsd1`, you want to find all the epochs for which the value is above 0.0. Pynapple provides the function `threshold` to get 1 dimensional time series above or below a certain value.
#
# **Question: Can you print the epochs for which `tsd1` is above 0.0?** 

ep_above = tsd1.threshold(0.0).time_support

print(ep_above)

# %%
# **Question**: can you plot `tsd1` as well as the epochs for which `tsd1` is above 0.0?

plt.figure()
plt.plot(tsd1)
plt.plot(tsd1.threshold(0.0), 'o-')
[plt.axvspan(s, e, alpha=0.2) for s,e in ep_above.values]
plt.show()



















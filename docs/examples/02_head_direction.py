# -*- coding: utf-8 -*-

"""
# Fit head-direction population

## Learning objectives

- Learn how to add history-related predictors to NeMoS GLM
- Learn about NeMoS `Basis` objects
- Learn how to use `Basis` objects with convolution

"""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import workshop_utils

import nemos as nmo

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(workshop_utils.STYLE_FILE)

# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.
# <div class="notes">
# - Stream the head-direction neurons data
# </div>

path = workshop_utils.fetch_data("Mouse32-140822.nwb")

# %%
# ## Pynapple
# We are going to open the NWB file with pynapple.
# <div class="notes">
# - `load_file` : open the NWB file and give a preview.
# </div>

data = nap.load_file(path)
data

# %%
#
# Get spike timings
# <div class="notes">
# - Load the units
# </div>

spikes = data["units"]
spikes

# %%
#
# Get the behavioural epochs (in this case, sleep and wakefulness)
#
# <div class="notes">
# - Load the epochs and take only wakefulness
# </div>

epochs = data["epochs"]
wake_ep = data["epochs"]["wake"]

# %%
# Get the tracked orientation of the animal
# <div class="notes">
# - Load the angular head-direction of the animal (in radians)
# </div>

angle = data["ry"]


# %%
# This cell will restrict the data to what we care about i.e. the activity of head-direction neurons during wakefulness.
#
# <div class="notes">
# - Select only those units that are in ADn
# - Restrict the activity to wakefulness (both the spiking activity and the angle)
# </div>

spikes = spikes.getby_category("location")["adn"]
spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)
angle = angle.restrict(wake_ep)

# %%
# First let's check that they are head-direction neurons.
# <div class="notes">
# - Compute tuning curves as a function of head-direction
# </div>

tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)

# %%
# Each row indicates an angular bin (in radians), and each column corresponds to a single unit.
# Let's plot the tuning curve of the first two neurons.

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tuning_curves.iloc[:, 0])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[1].plot(tuning_curves.iloc[:, 1])
ax[1].set_xlabel("Angle (rad)")
plt.tight_layout()

# %%
# Before using NeMoS, let's explore the data at the population level.
#
# Let's plot the preferred heading
#
# <div class="notes">
# - Let's visualize the data at the population level.
# </div>

fig = workshop_utils.plotting.plot_head_direction_tuning(
    tuning_curves, spikes, angle, threshold_hz=1, start=8910, end=8960
)

# %%
# As we can see, the population activity tracks very well the current head-direction of the animal.
#
# **Question : are neurons constantly tuned to head-direction and can we use it to predict the spiking activity of each neuron based only on the activity of other neurons?**
# 
# To fit the GLM faster, we will use only the first 3 min of wake
# <div class="notes">
# - Take the first 3 minutes of wakefulness to speed up optimization
# </div>

wake_ep = nap.IntervalSet(
    start=wake_ep.start[0], end=wake_ep.start[0] + 3 * 60
)

# %%
# To use the GLM, we need first to bin the spike trains. Here we use pynapple
# <div class="notes">
# - bin the spike trains in 10 ms bin
# </div>

bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

# %%
# Here we are going to rearrange neurons order based on their preferred directions.
#
# <div class="notes">
# - sort the neurons by their preferred direction using pandas
# </div>

pref_ang = tuning_curves.idxmax()

count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang.values)],
)

# %%
# ## NeMoS
# It's time to use NeMoS. Our goal is to estimate the pairwise interaction between neurons.
# This can be quantified with a GLM if we use the recent population spike history to predict the current time step.
# ### Self-Connected Single Neuron
# To simplify our life, let's see first how we can model spike history effects in a single neuron.
# The simplest approach is to use counts in fixed length window $i$, $y_{t-i}, \dots, y_{t-1}$ to predict the next
# count $y_{t}$. Let's plot the count history,
#
# <div class="notes">
# - Start with modeling a self-connected single neuron
# - Select a neuron and visualize the spike count time course
# </div>

# select a neuron's spike count time series
neuron_count = count[:, 0]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support.start[0], end=count.time_support.start[0] + 1.2
)
plt.figure(figsize=(8, 3.5))
plt.step(
    neuron_count.restrict(epoch_one_spk).t, neuron_count.restrict(epoch_one_spk).d, where="post"
)
plt.title("Spike Count Time Series")
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.tight_layout()

# %%
# #### Features Construction
# Let's fix the spike history window size that we will use as predictor.
# <div class="notes">
# - Use the past counts over a fixed window to predict the current sample
# </div>


# # set the size of the spike history window in seconds
window_size_sec = 0.8

workshop_utils.plotting.plot_history_window(neuron_count, epoch_one_spk, window_size_sec)


# %%
# For each time point, we shift our window one bin at the time and vertically stack the spike count history in a matrix.
# Each row of the matrix will be used as the predictors for the rate in the next bin (red narrow rectangle in
# the figure).
# <div class="notes">
# - Roll your window one bin at the time to predict the subsequent samples
# </div>

workshop_utils.plotting.run_animation(neuron_count, epoch_one_spk.start[0])

# %%
# If $t$ is smaller than the window size, we won't have a full window of spike history for estimating the rate.
# One may think of padding the window (with zeros for example) but this may generate weird border artifacts.
# To avoid that, we can simply restrict our analysis to times $t$ larger than the window and NaN-pad earlier
# time-points;
#
# A fast way to compute this feature matrix is convolving the counts with the identity matrix.
# We can apply the convolution and NaN-padding in a single step using the
# [`nemos.utils.create_convolutional_predictor`](../../../reference/nemos/utils/#nemos.utils.create_convolutional_predictor)
# function.
# <div class="notes">
# - Form a predictor matrix by vertically stacking all the windows (you can use a convolution).
# </div>

# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)

# convolve the counts with the identity matrix.
plt.close("all")
input_feature = nmo.convolve.create_convolutional_predictor(
    np.eye(window_size), neuron_count
)

# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0])

# %%
# The binned counts originally have shape "number of samples", we should check that the
# dimension are matching our expectation
# <div class="notes">
# - Check the shape of the counts and features.
# </div>

print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")
print(f"Feature shape: {input_feature.shape}")

# %%
#
# We can visualize the output for a few time bins
# <div class="notes">
# - Plot the convolution output.
# </div>

suptitle = "Input feature: Count History"
neuron_id = 0
workshop_utils.plotting.plot_features(input_feature, count.rate, suptitle)

# %%
# As you may see, the time axis is backward, this happens because convolution flips the time axis.
# This is equivalent, as we can interpret the result as how much a spike will affect the future rate.
# In the previous tutorial our feature was 1-dimensional (just the current), now
# instead the feature dimension is 80, because our bin size was 0.01 sec and the window size is 0.8 sec.
# We can learn these weights by maximum likelihood by fitting a GLM.

# %%
# #### Fitting the Model
#
# When working a real dataset, it is good practice to train your models on a chunk of the data and
# use the other chunk to assess the model performance. This process is known as "cross-validation".
# There is no unique strategy on how to cross-validate your model; What works best
# depends on the characteristic of your data (time series or independent samples,
# presence or absence of trials...), and that of your model. Here, for simplicity use the first
# half of the wake epochs for training and the second half for testing. This is a reasonable
# choice if the statistics of the neural activity does not change during the course of
# the recording. We will learn about better cross-validation strategies with other
# examples.
# <div class="notes">
# - Split your epochs in two for validation purposes.
# </div>

# construct the train and test epochs
duration = neuron_count.time_support.tot_length("s")
start = neuron_count.time_support["start"]
end = neuron_count.time_support["end"]
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)

# %%
# Fit the glm to the first half of the recording and visualize the ML weights.
# <div class="notes">
# - Fit a GLM to the first half.
# </div>


# define the GLM object
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))

# Fit over the training epochs
model.fit(
    input_feature.restrict(first_half),
    neuron_count.restrict(first_half)
)

# %%
# %%
# <div class="notes">
# - Plot the weights.
# </div>
plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_), lw=2, label="GLM raw history 1st Half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()

# %%
# The response in the previous figure seems noise added to a decay, therefore the response
# can be described with fewer degrees of freedom. In other words, it looks like we
# are using way too many weights to describe a simple response.
# If we are correct, what would happen if we re-fit the weights on the other half of the data?
# #### Inspecting the results
# <div class="notes">
# - Fit on the other half and compare results.
# </div>
# fit on the test set

model_second_half = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_second_half.fit(
    input_feature.restrict(second_half),
    neuron_count.restrict(second_half)
)

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_),
         label="GLM raw history 1st Half", lw=2)
plt.plot(np.arange(window_size) / count.rate,  np.squeeze(model_second_half.coef_),
         color="orange", label="GLM raw history 2nd Half", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()

# %%
# What can we conclude?
#
# The fast fluctuations are inconsistent across fits, indicating that
# they are probably capturing noise, a phenomenon known as over-fitting;
# On the other hand, the decaying trend is fairly consistent, even if
# our estimate is noisy. You can imagine how things could get
# worst if we needed a finer temporal resolution, such 1ms time bins
# (which would require 800 coefficients instead of 80).
# What can we do to mitigate over-fitting now?
#
# #### Reducing feature dimensionality
# One way to proceed is to find a lower-dimensional representation of the response
# by parametrizing the decay effect. For instance, we could try to model it
# with an exponentially decaying function $f(t) = \exp( - \alpha t)$, with
# $\alpha >0$ a positive scalar. This is not a bad idea, because we would greatly
# simplify the dimensionality our features (from 80 to 1). Unfortunately,
# there is no way to know a-priori what is a good parameterization. More
# importantly, not all the parametrizations guarantee a unique and stable solution
# to the maximum likelihood estimation of the coefficients (convexity).
#
# Let's see how to use NeMoS' `basis` module to reduce dimensionality instead!
# For history-type inputs, we'll use again the raised cosine log-stretched basis,
# [Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003).
# <div class="notes">
# - Visualize the raised cosine basis.
# </div>

workshop_utils.plotting.plot_basis()

# %%
# As before, we can instantiate this object in the `"conv"` mode of operation, and we
# can pass the window size for the convolution. With more basis functions, we'll be able to
# represent the effect of the corresponding input with the higher precision, at
# the cost of adding additional parameters.

# a basis object can be instantiated in "conv" mode for convolving  the input.
basis = nmo.basis.RaisedCosineBasisLog(
    n_basis_funcs=8, mode="conv", window_size=window_size
)

# time takes equi-spaced values between 0 and 1, we could multiply by the
# duration of our window to scale it to seconds.
time = window_size_sec * np.arange(window_size)

# %%
# Our spike history predictor was huge: every possible 80 time point chunk of the
# data, for 1440000 total numbers. By using this basis set we can instead reduce
# the predictor to 8 numbers for every 80 time point window for 144000 total
# numbers. Basically an order of magnitude less. With 1ms bins we would have
# achieved 2 order of magnitude reduction in input size. This is a huge benefit
# in terms of memory allocation and, computing time. As an additional benefit,
# we will reduce over-fitting.
#
# Let's see our basis in action. We can "compress" spike history feature by convolving the basis
# with the counts (without creating the large spike history feature matrix).
# This can be performed in NeMoS by calling the `compute_features` method of basis.
#
# <div class="notes">
# - Convolve the counts with the basis functions.
# </div>

# equivalent to
# `nmo.convolve.create_convolutional_predictor(basis_kernels, neuron_count)`
conv_spk = basis.compute_features(neuron_count)

print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")

# %%
# <div class="notes">
# - Visualize the output.
# </div>

# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

workshop_utils.plotting.plot_convolved_counts(neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk)

# %%
# Now that we have our "compressed" history feature matrix, we can fit the ML parameters for a GLM.
#
# #### Fit and compare the models
# <div class="notes">
# - Fit the model using the compressed features.
# </div>
# use restrict on interval set training
model_basis = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis.fit(conv_spk.restrict(first_half), neuron_count.restrict(first_half))

# %%
# We can plot the resulting response, noting that the weights we just learned needs to be "expanded" back
# to the original `window_size` dimension by multiplying them with the basis kernels.
# We have now 8 coefficients,

print(model_basis.coef_)

# %%
# In order to get the response we need to multiply the coefficients by their corresponding
# basis function, and sum them.
# <div class="notes">
# - Reconstruct the history filter.
# </div>
_, basis_kernels = basis.evaluate_on_grid(window_size)
self_connection = np.matmul(basis_kernels, np.squeeze(model_basis.coef_))

print(self_connection.shape)

# %%
# We can now compare this model that based on the raw count history.
# %%
# <div class="notes">
# - Compare with the raw count history model.
# </div>
plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), alpha=0.3, label="GLM raw history")
plt.plot(time, self_connection, "--k", label="GLM basis", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()

# %%
# Let's check if our new estimate does a better job in terms of over-fitting. We can do that
# by visual comparison, as we did previously. Let's fit the second half of the dataset.
#
#
# <div class="notes">
# - Fit the other half of the data.
# - Plot and compare the results.
# </div>
model_basis_second_half = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis_second_half.fit(conv_spk.restrict(second_half), neuron_count.restrict(second_half))

# compute responses for the 2nd half fit
self_connection_second_half = np.matmul(basis_kernels, np.squeeze(model_basis_second_half.coef_))

plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), "k", alpha=0.3, label="GLM raw history 1st half")
plt.plot(time, np.squeeze(model_second_half.coef_), alpha=0.3, color="orange", label="GLM raw history 2nd half")
plt.plot(time, self_connection, "--k", lw=2, label="GLM basis 1st half")
plt.plot(time, self_connection_second_half, color="orange", lw=2, ls="--", label="GLM basis 2nd half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()

# %%
# Let's extract and plot the rates
# <div class="notes">
# - Predict the rates and plot the results.
# </div>
rate_basis = model_basis.predict(conv_spk) * conv_spk.rate
rate_history = model.predict(input_feature) * conv_spk.rate
ep = nap.IntervalSet(start=8819.4, end=8821)

# plot the rates
workshop_utils.plotting.plot_rates_and_smoothed_counts(
    neuron_count.restrict(ep),
    {"Self-connection raw history":rate_history, "Self-connection bsais": rate_basis}
)

# %%
# ### All-to-all Connectivity
# The same approach can be applied to the whole population. Now the firing rate of a neuron
# is predicted not only by its own count history, but also by the rest of the
# simultaneously recorded population. We can convolve the basis with the counts of each neuron
# to get an array of predictors of shape, `(num_time_points, num_neurons * num_basis_funcs)`.
#
# #### Preparing the features
# <div class="notes">
# - Convolve all counts.
# - Print the output shape
# </div>
# convolve all the neurons
convolved_count = basis.compute_features(count)

# %%
# Check the dimension to make sure it make sense
# Shape should be (n_samples, n_basis_func * n_neurons)
print(f"Convolved count shape: {convolved_count.shape}")

# %%
# #### Fitting the Model
# This is an all-to-all neurons model.
# We are using the class `PopulationGLM` to fit the whole population at once.
#
# !!! note
#     Once we condition on past activity, log-likelihood of the population is the sum of the log-likelihood
#     of individual neurons. Maximizing the sum (i.e. the population log-likelihood) is equivalent to
#     maximizing each individual term separately (i.e. fitting one neuron at the time).
#
#
# <div class="notes">
# - Fit a `PopulationGLM`
# </div>

model = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"),
    ).fit(convolved_count, count)

# %%
# #### Comparing model predictions.
# Predict the rate (counts are already sorted by tuning prefs)
# <div class="notes">
# - Predict the firing rate of each neuron
# - Convert the rate from spike/bin to spike/sec
# </div>
predicted_firing_rate = model.predict(convolved_count) * conv_spk.rate

# %%
# Plot fit predictions over a short window not used for training.
#
# <div class="notes">
# - Visualize the predicted rate and tuning function.
# </div>

# use pynapple for time axis for all variables plotted for tick labels in imshow
workshop_utils.plotting.plot_head_direction_tuning_model(tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
                                          start=8910, end=8960, cmap_label="hsv")
# %%
# Let's see if our firing rate predictions improved and in what sense.
# <div class="notes">
# - Visually compare all the models.
# </div>

# mkdocs_gallery_thumbnail_number = 2
workshop_utils.plotting.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: bsais": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)

# %%
# #### Visualizing the connectivity
# Compute the tuning curve form the predicted rates.
# <div class="notes">
# - Compute tuning curves from the predicted rates using pynapple.
# </div>
tuning = nap.compute_1d_tuning_curves_continuous(predicted_firing_rate,
                                                 feature=angle,
                                                 nb_bins=61,
                                                 minmax=(0, 2 * np.pi))

# %%
# Extract the weights and store it in a (n_neurons, n_neurons, n_basis_funcs) array.
# <div class="notes">
# - Extract the weights and store it in an array,
#   shape (num_neurons, num_neurons, num_features).
# </div>

weights = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])


# %%
# Multiply the weights by the basis, to get the history filters.
# <div class="notes">
# - Multiply the weights by the basis, to get the history filters.
# </div>

responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

print(responses.shape)

# %%
# Finally, we can visualize the pairwise interactions by plotting
# all the coupling filters.
# <div class="notes">
# - Plot the connectivity map.
# </div>

workshop_utils.plotting.plot_coupling(responses, tuning)

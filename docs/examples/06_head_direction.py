# -*- coding: utf-8 -*-

"""
# Fit head-direction population

## Learning objectives {.keep-text}

- Learn how to add history-related predictors to NeMoS GLM
- Learn how to reduce over-fitting with `Basis`
- Learn how to cross-validate with NeMoS + scikit-learn

"""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import warnings
import workshop_utils

import nemos as nmo
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

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
# ## NeMoS {.strip-code}
# It's time to use NeMoS. Our goal is to estimate the pairwise interaction between neurons.
# This can be quantified with a GLM if we use the recent population spike history to predict the current time step.
# ### Self-Connected Single Neuron
# To simplify our life, let's see first how we can model spike history effects in a single neuron.
# We can follow the simplest approach at first, which is to
# use counts in fixed length window $i$, $y_{t-i}, \dots, y_{t-1}$ to predict the next
# count $y_{t}$.
#
# <div class="notes">
# - Start with modeling a self-connected single neuron
# - Select a neuron
# - Select the first 1.2 seconds for visualization
# </div>

# select a neuron's spike count time series
neuron_count = count[:, 0]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support.start[0], end=count.time_support.start[0] + 1.2
)

# %%
# Plot the window.
# <div class="notes">
# - visualize the spike count time course
# </div>

# {.keep-code}
# set the size of the spike history window in seconds
window_size_sec = 0.8

workshop_utils.plotting.plot_history_window(
    neuron_count, epoch_one_spk, window_size_sec
)

# %%
# The feature matrix is obtained by shifting the prediction window forward in time and stacking vertically
# the result in an array of shape `(n_shift, window_size)`.
# A fast wat to do so is convolving the counts with the identity matrix.
# <div class="notes">
# - Form a predictor matrix by vertically stacking all the windows (you can use a convolution).
# </div>

# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)

# convolve the counts with the identity matrix.
input_feature = nmo.convolve.create_convolutional_predictor(
    np.eye(window_size), neuron_count
)

# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0])

# %%
# We should check that the dimension are matching our expectation.
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

# {.keep-code}
suptitle = "Input feature: Count History"
neuron_id = 0
workshop_utils.plotting.plot_features(input_feature, count.rate, suptitle)

# %%
# As you may see, the time axis is backward, this happens because convolution flips the time axis.
# This is equivalent, as we can interpret the result as how much a spike will affect the future rate.
# The resulting feature dimension is 80, because our bin size was 0.01 sec and the window size is 0.8 sec.
# We can learn these weights by maximum likelihood by fitting a GLM.

# %%
# #### Fitting the Model
#
# When working with a real dataset, it is good practice to train your models on a chunk of the data and
# use the other chunk to assess the model performance. This process is known as **"cross-validation"**.
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

# {.keep-code}
# construct the train and test epochs
duration = neuron_count.time_support.tot_length()
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
model = nmo.glm.GLM()

# Fit over the training epochs
model.fit(
    input_feature.restrict(first_half),
    neuron_count.restrict(first_half)
)

# %%
# <div class="notes">
# - Plot the weights.
# </div>

# {.keep-code}
workshop_utils.plotting.plot_and_compare_weights(
    [model.coef_], ["GLM raw history 1st Half"], count.rate)

# %%
# The response in the previous figure seems noise added to a decay, therefore the response
# can be described with fewer degrees of freedom. In other words, it looks like we
# are using way too many weights to describe a simple response.
# If we are correct, what would happen if we re-fit the weights on the other half of the data?
# #### Inspecting the results
# <div class="notes">
# - Fit on the other half.
# </div>
# Fit on the test set.

model_second_half = nmo.glm.GLM()
model_second_half.fit(
    input_feature.restrict(second_half),
    neuron_count.restrict(second_half)
)

# %%
# Compare the results
# <div class="notes">
# - Compare results.
# </div>

# {.keep-code}
workshop_utils.plotting.plot_and_compare_weights(
    [model.coef_, model_second_half.coef_],
    ["GLM raw history 1st Half", "GLM raw history 2nd Half"],
    count.rate)

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
# Let's see how to use NeMoS' `basis` module to reduce dimensionality and avoid over-fitting!
# For history-type inputs, we'll use again the raised cosine log-stretched basis,
# [Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003).
# <div class="notes">
# - Visualize the raised cosine basis.
# </div>

# {.keep-code}
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

# {.keep-code}
# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

workshop_utils.plotting.plot_convolved_counts(
    neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk
)

# %%
# Now that we have our "compressed" history feature matrix, we can fit the ML parameters for a GLM.
#
# #### Fit and compare the models
# <div class="notes">
# - Fit the model using the compressed features.
# </div>


model_basis = nmo.glm.GLM()
# use restrict on interval set training
model_basis.fit(
    conv_spk.restrict(first_half),
    neuron_count.restrict(first_half)
)

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
self_connection = np.matmul(basis_kernels, model_basis.coef_)

print(self_connection.shape)

# %%
# Let's check if our new estimate does a better job in terms of over-fitting. We can do that
# by visual comparison, as we did previously. Let's fit the second half of the dataset.
#
#
# <div class="notes">
# - Fit the other half of the data.
# </div>
model_basis_second_half = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS")
)
model_basis_second_half.fit(
    conv_spk.restrict(second_half), neuron_count.restrict(second_half)
)

# compute responses for the 2nd half fit
self_connection_second_half = np.matmul(basis_kernels, model_basis_second_half.coef_)

# %%
# We can now compare this model that based on the raw count history.
#
# <div class="notes">
# - Plot and compare the results.
# </div>

workshop_utils.plotting.plot_and_compare_weights(
    [model.coef_, model_second_half.coef_, self_connection, self_connection_second_half],
    ["GLM raw history 1st Half", "GLM raw history 2nd half", "GLM basis 1st half", "GLM basis 2nd half"],
    count.rate
)


# %%
# Let's extract and plot the rates
# <div class="notes">
# - Predict the rates
# </div>
rate_basis = model_basis.predict(conv_spk) * conv_spk.rate
rate_history = model.predict(input_feature) * conv_spk.rate
ep = nap.IntervalSet(start=8819.4, end=8821)

# %%
# <div class="notes">
# - Plot the results.
# </div>

# {.keep-code}
# plot the rates
workshop_utils.plotting.plot_rates_and_smoothed_counts(
    neuron_count.restrict(ep),
    {
        "Self-connection raw history":rate_history,
        "Self-connection bsais": rate_basis
    }
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
# Check the dimension to make sure it make sense.

# shape should be `(n_samples, n_basis_func * n_neurons)`
print(f"Convolved count shape: {convolved_count.shape}")

# %%
# #### Fitting the Model
# This is an all-to-all neurons model.
# We can use the class `PopulationGLM` to fit the whole population at once.
#
# How many weights are we learning in this case? We have 8 x 19 = 152 features for each of our 19 neurons,
# for a total of 2888 weights, so the parameter space is still quite large.
# A safe approach to further mitigate over-fitting is to use a Ridge (L2) penalization.
#
# !!! note
#     Once we condition on past activity, log-likelihood of the population is the sum of the log-likelihood
#     of individual neurons. Maximizing the sum (i.e. the population log-likelihood) is equivalent to
#     maximizing each individual term separately (i.e. fitting one neuron at the time).
#
#
# <div class="notes">
# - Fit a `PopulationGLM`
# - Use Ridge regularization with a `regularizer_strength=0.1`
# - Print the shape of the estimated coefficients.
# </div>


model = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.Ridge("LBFGS", regularizer_strength=0.1)
).fit(convolved_count, count)

print(f"Model coefficients shape: {model.coef_.shape}")

# %%
# #### Comparing model predictions.
# <div class="notes">
# - Predict the firing rate of each neuron
# - Convert the rate from spike/bin to spike/sec
# </div>

# predict the rate (counts are already sorted by tuning prefs)
predicted_firing_rate = model.predict(convolved_count) * conv_spk.rate

# %%
# Plot fit predictions over a short window not used for training.
#
# <div class="notes">
# - Visualize the predicted rate and tuning function.
# </div>

# {.keep-code}
# use pynapple for time axis for all variables plotted for tick labels in imshow
workshop_utils.plotting.plot_head_direction_tuning_model(
    tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
    start=8910, end=8960, cmap_label="hsv"
)
# %%
# Let's see if our firing rate predictions improved and in what sense.
# <div class="notes">
# - Visually compare all the models.
# </div>

# {.keep-code}
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

n_neurons = count.shape[1]
weights = model.coef_.reshape(n_neurons, basis.n_basis_funcs, n_neurons)


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

# {.keep-code}
workshop_utils.plotting.plot_coupling(responses, tuning)


# %%
# ## K-fold Cross-Validation {.keep-paragraph}
#
# <p align="center">
#   <img src="../../../assets/grid_search_cross_validation.png" alt="Grid Search Cross Validation" style="max-width: 80%; height: auto;">
#   <br>
#   <em>K-fold cross-validation (from <a href="https://scikit-learn.org/stable/modules/cross_validation.html" target="_blank">scikit-learn docs</a>)</em>
# </p>
#
#
# Here, we selected a reasonable regularization strength for the Ridge-GLM for you.
# In general, figuring out a "good" value for this hyperparameter is crucial for model fit quality.
# Too low, and you may over-fit (high variance), too high, and you may
# under-fit (high bias), i.e. learning very small weights that do not capture neural activity.
#
# What you aim for is to strike a balance between the variance and the bias. Quantitatively, you can assess
# how well your model is performing by evaluating the log-likelihood score over some
# left-out data.
#
# A common approach is the "K-fold" cross validation, see figure above.
# In a K-fold cross-validation, you'll split the data in K chunks of equal size. You then fit the
# model on K-1 chunks, and score it on the left-out one.
# You'll repeat the procedure K-times leaving out a different chunk at each iteration.
# At the end of the procedure, you can average the K-scores, to get a robust estimate of the model
# performance.
# To select a hyperparameter, you can run a K-fold over a grid of hyperparameters,
# and pick the one with the best average score.
#
# ## K-fold with NeMoS and scikit-learn {.strip-code}
# Let's see how to implement the K-Fold with NeMoS and scikit-learn.
#
# <div class="notes">
# - Instantiate the PopulationGLM
# - Define a grid of regularization strengths.
# - Instantiate and fit the GridSearchCV with 2 folds.
# </div>

# {.keep-code}
# define the model
model = nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.Ridge("LBFGS")
)

# define a grid of parameters for the search
param_grid = dict(regularizer__regularizer_strength=np.logspace(-3, 0, 4))
param_grid

# define a GridSearch cross-validation from scikit-learn
# with 2-folds
k_fold = GridSearchCV(model, param_grid=param_grid, cv=2)

# %%
# !!! note
#
#     The keys in `param_grid` use a special syntax of the form
#     `<parameter>__<subparameter>`. This tells scikit-learn to access and set the
#     values of the `model.parameter.subparameter` attribute.
#
#     See the [scikit-learn
#     docs](https://scikit-learn.org/stable/modules/grid_search.html#composite-estimators-and-parameter-spaces)
#     for more details.
#
# <div class="notes">
# - Run cross-validation!
# </div>

# {.keep-code}
# fit the cross-validated model
k_fold.fit(convolved_count, count)

# %%
# We can inspect the K-fold result and print best parameters.
# <div class="notes">
# - Print the best parameters.
# </div>

# {.keep-code}
print(f"Best regularization strength: "
      f"{k_fold.best_params_['regularizer__regularizer_strength']}")

# %%
# ## Exercises {.keep-text}
# - Plot the weights and rate predictions.
# - What happens if you use 5 folds?
# - What happen if you cross-validate each neuron individually?
#   Do you select the same hyperparameter for every neuron or not?



# -*- coding: utf-8 -*-

"""
# Combining and comparing models.


The data for this example are from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from workshop_utils import fetch_data, plotting
from scipy.ndimage import gaussian_filter

import nemos as nmo


# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.

path = fetch_data("Achilles_10252013.nwb")

# %%
# ## Pynapple
# We are going to open the NWB file with pynapple

data = nap.load_file(path)

print(data)

# %%
# Let's extract the spike times, the position and the theta phase.

spikes = data["units"]
position = data["position"]
theta = data["theta_phase"]

# %%
# The NWB file also contains the time at which the animal was traversing the linear track. We can use it to restrict the position and assign it as the `time_support` of position.

position = position.restrict(data["trials"])

# %%
# The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells. Neurons have already been labelled before.
spikes = spikes.getby_category("cell_type")["pE"]

# %%
# We can discard the low firing neurons as well.
spikes = spikes.getby_threshold("rate", 0.3)

# %%
# ## Place fields
# Let's plot some data. We start by making place fields i.e. firing rate as a function of position.

pf = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)

# %%
# Let's do a quick sort of the place fields for display
order = pf.idxmax().sort_values().index.values

# %%
# Here each row is one neuron

plt.figure(figsize=(12, 10))
gs = plt.GridSpec(len(spikes), 1)
for i, n in enumerate(order):
    plt.subplot(gs[i, 0])
    plt.fill_between(pf.index.values, np.zeros(len(pf)), pf[n].values)
    if i < len(spikes) - 1:
        plt.xticks([])
    else:
        plt.xlabel("Position (cm)")
    plt.yticks([])


# %%
# ## Phase precession
#
# In addition to place modulation, place cells are also modulated by the theta oscillation. The phase at which neurons fire is dependant of the position. This phenomen is called "phase precession" (see [J. O’Keefe, M. L. Recce, "Phase relationship between hippocampal place units and the EEG theta rhythm." Hippocampus 3, 317–330 (1993).](https://doi.org/10.1002/hipo.450030307)
#
# Let's compute the response of the neuron as a function of both theta and position. The phase of theta has already been computed but we have to bring it to the same dimension as the position feature. While the position has been sampled at 40Hz, the theta phase has been computed at 1250Hz.
# Later on during the analysis, we will use a bin size of 5 ms for counting the spikes. Since this corresponds to an intermediate frequency between 40 and 1250 Hz, we will bring all the features to 200Hz already.

bin_size = 0.005

theta = theta.bin_average(bin_size, position.time_support)
theta = (theta + 2 * np.pi) % (2 * np.pi)

data = nap.TsdFrame(
    t=theta.t,
    d=np.vstack(
        (position.interpolate(theta, ep=position.time_support).values, theta.values)
    ).T,
    time_support=position.time_support,
    columns=["position", "theta"],
)

print(data)


# %%
# `data` is a `TsdFrame` that contains the position and phase. Before calling `compute_2d_tuning_curves` from pynapple to observe the theta phase precession, we will restrict the analysis to the place field of one neuron.
#
# There are a lot of neurons but for this analysis, we will focus on one neuron only.
neuron = 175

plt.figure(figsize=(5,3))
plt.fill_between(pf[neuron].index.values, np.zeros(len(pf)), pf[neuron].values)
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")

# %%
# This neurons place field is between 0 and 60 cm within the linear track. Here we will use the `threshold` function of pynapple to quickly compute the epochs for which the animal is within the place field :

within_ep = position.threshold(60.0, method="below").time_support

# %%
# `within_ep` is an `IntervalSet`. We can now give it to `compute_2d_tuning_curves` along with the spiking activity and the position-phase features.

tc_pos_theta, xybins = nap.compute_2d_tuning_curves(spikes, data, 20, within_ep)

# %%
# To show the theta phase precession, we can also display the spike as a function of both position and theta. In this case, we use the function `value_from` from pynapple.

theta_pos_spikes = spikes[neuron].value_from(data, ep = within_ep)

# %%
# Now we can plot everything together :

plt.figure()
gs = plt.GridSpec(2, 2)
plt.subplot(gs[0, 0])
plt.fill_between(pf[neuron].index.values, np.zeros(len(pf)), pf[neuron].values)
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")

plt.subplot(gs[1, 0])
extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])
plt.imshow(gaussian_filter(tc_pos_theta[neuron].T, 1), aspect="auto", origin="lower", extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.subplot(gs[1, 1])
plt.plot(theta_pos_spikes["position"], theta_pos_spikes["theta"], "o", markersize=0.5)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.tight_layout()

# %%
# ## Speed modulation
# The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze.
# In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

speed = []
for s, e in data.time_support.values: # Time support contains the epochs
    pos_ep = data["position"].get(s, e)
    speed_ep = np.abs(np.diff(pos_ep)) # Absolute difference of two consecutive points
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") # Adding one point at the end to match the size of the position array
    speed_ep = speed_ep * data.rate # Converting to cm/s
    speed.append(speed_ep)

speed = nap.Tsd(t=data.t, d=np.hstack(speed), time_support=data.time_support)

# %%
# Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple `compute_1d_tuning_curves`:

tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20)

# %%
# To assess the variabilty in speed when the animal is travering the linear track, we can compute the average speed and estimate the standard deviation. Here we use numpy only and put the results in a pandas `DataFrame`:

bins = np.linspace(np.min(data["position"]), np.max(data["position"]), 20)

idx = np.digitize(data["position"].values, bins)

mean_speed = np.array([np.mean(speed[idx==i]) for i in np.unique(idx)])
std_speed = np.array([np.std(speed[idx==i]) for i in np.unique(idx)])

# %%
# Here we plot the tuning curve of one neuron for speed as well as the average speed as a function of the animal position

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.plot(bins, mean_speed)
plt.fill_between(
    bins,
    mean_speed - std_speed,
    mean_speed + std_speed,
    alpha=0.1,
)
plt.xlabel("Position (cm)")
plt.ylabel("Speed (cm/s)")
plt.title("Animal speed")
plt.subplot(122)
plt.fill_between(
    tc_speed.index.values, np.zeros(len(tc_speed)), tc_speed[neuron].values
)
plt.xlabel("Speed (cm/s)")
plt.ylabel("Firing rate (Hz)")
plt.title("Neuron {}".format(neuron))
plt.tight_layout()

# %%
# This neurons show a strong modulation of firing rate as a function of speed but we can also notice that the animal, on average, accelerates when travering the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speed and for different theta phase? We can use NeMoS to model the activity and give the position, the phase and the speed as input variable.
#
# We will use speed, phase and position to model the activity of the neuron.
# All the feature have already been brought to the same dimension thanks to `pynapple`.

position = data["position"]
theta = data["theta"]
count = spikes[neuron].count(bin_size, data.time_support)

print(position.shape)
print(theta.shape)
print(speed.shape)
print(count.shape)

# %%
# ## Basis evaluation {.strip-code,.keep-text}
#
# There are multiple features for this experiment that can explain the spiking activity i.e. `position`, `theta` and `speed`.
#
# **Question:**Can you instantiate the right basis for each feature and call them respectively `position_basis`, `theta_basis`, `speed_basis`?

position_basis = nmo.basis.MSplineBasis(n_basis_funcs=10)
phase_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
speed_basis = nmo.basis.MSplineBasis(n_basis_funcs=15)

# %%
#
# Hippocampal place cells fires both in at a preferential phase for a particular position as seen above. To model this interaction, you need to combine basis with NeMoS.
#
# **Question:** Can you crate a new basis that is the product of `phase_basis` and `position_basis`?

position_phase_basis = position_basis * phase_basis

# %%
# This basis set can be used to generate a design matrix.
# 
# **Question:** Using the right features, can you generate the right design matrix from the basis defined above?

X1 = position_phase_basis(position, theta)

print(X1)

# %%
# `X1` is our design matrix. It's time to learn our first model.
#
# ## Model learning {.strip-code,.keep-text}
#
# **Question:** Can you instantiate an unregularized GLM class with `LBFGS` as a solver?

glm1 = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS", solver_kwargs=dict(tol=10**-12))
)

# %%
# Let's reserve half of the epochs for training and half is going to be use to compare model (testing).

ep_training = count.time_support[::2]
ep_testing = count.time_support[1::2]

# %%
# **Question:** ... and fit the model?

glm1.fit(X1.restrict(ep_training), count.restrict(ep_training))

# %%
# ## Prediction {.strip-code,.keep-text}
#
# It's time to predict some activity and see if we capture the position and phase interaction.
# 
# **Question:** Using the `predict` function of NeMoS, can you compute the firing in spikes per second?

pred_rate_1 = glm1.predict(X1.restrict(ep_training))/bin_size

# %%
# We can compute a tuning curves from the predicted rate.
#
# **Question:** Using the right pynapple function, can you compute a 2D tuning curves of "phase x position" can call it `glm1_pos_theta`?
#

glm1_position_phase, xybins = nap.compute_2d_tuning_curves_continuous(
    pred_rate_1, data, 30, ep=within_ep
)

# %%
# We can compare this to the observed phase-position tuning curves.

# {.keep-code}

extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])

plt.figure(figsize = (15,4))
plt.subplot(121)
plt.title("Raw Tuning")
plt.imshow(gaussian_filter(tc_pos_theta[neuron].T, 1), aspect="auto", origin="lower", extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.subplot(122)
plt.title("GLM 1 predicted Tuning")
plt.imshow(gaussian_filter(np.transpose(glm1_position_phase[0]), 1), aspect="auto", origin="lower", extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.tight_layout()

# %%
# While this looks good, we can look at position and speed individually.
#
# **Question:** Using the right pynapple function, can you compute 1D tuning curves for `position` and `speed`

glm1_position = nap.compute_1d_tuning_curves_continuous(pred_rate_1, position, 50)
glm1_speed = nap.compute_1d_tuning_curves_continuous(pred_rate_1, speed, 30, minmax=(0, 100))

# %%
# ... and we can plot them next to the original tuning curves?

# {.keep-code}
plt.figure(figsize = (15,4))

plt.subplot(121)
plt.title("position")
plt.ylabel("rate (Hz)")
plt.plot(pf[neuron])
plt.plot(glm1_position)

plt.xlabel("cm")

plt.subplot(122)
plt.title("speed")
plt.plot(tc_speed[neuron])
plt.plot(glm1_speed)
plt.xlabel("cm/sec")

# %%
# **Question:** Even if we did not include explicitly speed as a regression we can capture some tuning. Why is that?
#
# Let's include the speed as a predictor to see if we get a qualitatively better match.
#
# **Question:** Can you define a new basis that captures the effect of speed? *(tip: add an extra basis)*

basis = position_phase_basis + speed_basis

# %%
# Let's use the basis to create a new design matrix and fit a GLM on the training epoch.

X2 = basis(position, theta, speed)

glm2 = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS", solver_kwargs=dict(tol=10**-12))
)

# fit
glm2.fit(X2.restrict(ep_training), count.restrict(ep_training))

# %%
# Let's compute again the tuning function from this model using pynapple.

# predict rate
pred_rate_2 = glm2.predict(X2.restrict(ep_training))/bin_size

# compute 1d and 2d tuning
glm2_position_phase, xybins = nap.compute_2d_tuning_curves_continuous(
    pred_rate_2, data, 30, ep=within_ep
)

glm2_position = nap.compute_1d_tuning_curves_continuous(pred_rate_2, position, 50)
glm2_speed = nap.compute_1d_tuning_curves_continuous(pred_rate_2, speed, 30, minmax=(0, 100))


# %%
# **Question:** if you re-plot the tuning function for this model, does it look like it is doing a better job
# on capturing the speed tuning?

plt.figure()

plt.subplot(221)
plt.title("position")
plt.ylabel("rate (Hz)")
plt.plot(pf[neuron])
plt.plot(glm1_position, label="position x phase")
plt.plot(glm2_position, label="position x phase + speed")
plt.legend()
plt.xlabel("cm")

plt.subplot(222)
plt.title("speed")
plt.plot(tc_speed[neuron])
plt.plot(glm1_speed, label="position x phase")
plt.plot(glm2_speed, label="position x phase + speed")
plt.xlabel("cm/sec")
plt.legend()
plt.tight_layout()

# %%
# How do we make this quantitative?
# **Question:** can you use the `score` method of `GLM` to check which model has a better likelihood on the test epochs?

print(f"position x phase score: {glm1.score(X1.restrict(ep_testing), count.restrict(ep_testing))}")
print(f"position x phase + speed score: {glm2.score(X2.restrict(ep_testing), count.restrict(ep_testing))}")

# %%
# ## Conclusion
#
# Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :
#
#   - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)
#
#   - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)
#
#   - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)
#

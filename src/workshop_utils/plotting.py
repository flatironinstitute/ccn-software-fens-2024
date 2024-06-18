#!/usr/bin/env python3

from typing import Optional

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


def tuning_curve_plot(tuning_curve: pd.DataFrame):
    fig, ax = plt.subplots(1, 1)
    tc_idx = tuning_curve.index.to_numpy()
    tc_val = tuning_curve.values.flatten()
    width = tc_idx[1]-tc_idx[0]
    ax.bar(tc_idx, tc_val, width, facecolor="grey", edgecolor="k",
           label="observed", alpha=0.4)
    ax.set_xlabel("Current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    return fig


def current_injection_plot(current: nap.Tsd, spikes: nap.TsGroup,
                           firing_rate: nap.TsdFrame,
                           *predicted_firing_rates: Optional[nap.TsdFrame]):
    ex_intervals = current.threshold(0.0).time_support

    # define plotting parameters
    # colormap, color levels and transparency level
    # for the current injection epochs
    cmap = plt.get_cmap("autumn")
    color_levs = [0.8, 0.5, 0.2]
    alpha = 0.4

    fig = plt.figure(figsize=(7, 7))
    # first row subplot: current
    ax = plt.subplot2grid((4, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(current, color="grey")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Injected Current")
    ax.set_xticklabels([])
    ax.axvspan(ex_intervals.loc[0,"start"], ex_intervals.loc[0,"end"], alpha=alpha, color=cmap(color_levs[0]))
    ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[1]))
    ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[2]))

    # second row subplot: response
    resp_ax = plt.subplot2grid((4, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
    resp_ax.plot(firing_rate, color="k", label="Observed firing rate")
    if predicted_firing_rates:
        if len(predicted_firing_rates) > 1:
            lbls = [' (current history)', ' (instantaneous only)']
        else:
            lbls = ['']
        for pred_fr, style, lbl in zip(predicted_firing_rates, ['-', '--'], lbls):
            resp_ax.plot(pred_fr, linestyle=style, color="tomato", label=f'Predicted firing rate{lbl}')
    resp_ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10, label="Observed spikes")
    resp_ax.set_ylabel("Firing rate (Hz)")
    resp_ax.set_xlabel("Time (s)")
    resp_ax.set_title("Neural response", y=.95)
    resp_ax.axvspan(ex_intervals.loc[0,"start"], ex_intervals.loc[0,"end"], alpha=alpha, color=cmap(color_levs[0]))
    resp_ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[1]))
    resp_ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[2]))
    ylim = resp_ax.get_ylim()

    # third subplot: zoomed responses
    zoom_axes = []
    for i in range(len(ex_intervals)):
        interval = ex_intervals.loc[[i]]
        ax = plt.subplot2grid((4, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
        ax.plot(firing_rate.restrict(interval), color="k")
        ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
        if predicted_firing_rates:
            for pred_fr, style in zip(predicted_firing_rates, ['-', '--']):
                ax.plot(pred_fr.restrict(interval), linestyle=style,
                        color="tomato")
        else:
            ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        for spine in ["left", "right", "top", "bottom"]:
            color = cmap(color_levs[i])
            # add transparency
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)
        zoom_axes.append(ax)

    resp_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.4),
                   bbox_transform=zoom_axes[1].transAxes)


def plot_head_direction_tuning(
        tuning_curves: pd.DataFrame,
        spikes: nap.TsGroup,
        angle: nap.Tsd,
        threshold_hz: int = 1,
        start: float = 8910,
        end: float = 8960,
        cmap_label="hsv",
        figsize=(12, 6)
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:

    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------

    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (unq_angles[-1] - unq_angles[0])
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid((3, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid((3, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(spike_tsd[sel].t, np.ones(sel.sum()) * i, "|", color=cmap(relative_color_levs[i]), alpha=0.5)
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid((3, n_subplots), loc=(2 + i // n_subplots, i % n_subplots),
                              rowspan=1, colspan=1, fig=fig, projection="polar")
        ax.fill_between(tuning_curves.iloc[:, neu_idx].index, np.zeros(len(tuning_curves)),
                        tuning_curves.iloc[:, neu_idx].values, color=cmap(relative_color_levs[i]), alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()


def plot_head_direction_tuning_model(
        tuning_curves: pd.DataFrame,
        predicted_firing_rate: nap.TsdFrame,
        spikes: nap.TsGroup,
        angle: nap.Tsd,
        threshold_hz: int = 1,
        start: float = 8910,
        end: float = 8960,
        cmap_label="hsv",
        figsize=(12, 6)
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:
        The tuning curve dataframe.
    predicted_firing_rate:
        The time series of the predicted rate.
    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------
    fig:
        The figure.
    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = (
        spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)
    )

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (unq_angles[-1] - unq_angles[0])
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid(
        (4, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid(
        (4, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(
            spike_tsd[sel].t,
            np.ones(sel.sum()) * i,
            "|",
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    ax = plt.subplot2grid(
        (4, n_subplots), loc=(2, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Firing Rate")

    fr = predicted_firing_rate.restrict(plot_ep).d
    fr = fr.T / np.max(fr, axis=1)
    ax.imshow(fr[::-1], cmap="Blues", aspect="auto")
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid(
            (4, n_subplots),
            loc=(3 + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves.iloc[:, neu_idx].index,
            np.zeros(len(tuning_curves)),
            tuning_curves.iloc[:, neu_idx].values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()


def plot_features(
        input_feature: NDArray,
        sampling_rate: float,
        suptitle:str,
        n_rows: int = 20
):
    """
    Plot feature matrix.

    Parameters
    ----------
    input_feature:
        The (num_samples, n_neurons, num_feature) feature array.
    sampling_rate:
        Sampling rate in hz.
    n_rows:
        Number of rows to plot.
    suptitle:
        Suptitle of the plot.

    Returns
    -------

    """
    window_size = input_feature.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    time = np.arange(0, window_size) / sampling_rate
    input_feature = input_feature.dropna()
    for k in range(n_rows):
        ax = plt.subplot(n_rows, 1, k + 1)
        plt.step(time, input_feature[k], where="post")

        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.axvspan(0, time[-1], alpha=0.4, color="orange")
        ax.set_yticks([])
        if k != n_rows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("lag (sec)")
        if k in [0, n_rows - 1]:
            ax.set_ylabel("$t_{%d}$" % (window_size + k), rotation=0)

    plt.tight_layout()


def plot_coupling(responses, tuning, cmap_name="seismic",
                      figsize=(10, 8), fontsize=15, alpha=0.5, cmap_label="hsv"):
    pref_ang = tuning.idxmax()
    cmap_tun = plt.get_cmap(cmap_label)
    color_tun = (pref_ang.values - pref_ang.values.min()) / (pref_ang.values.max() - pref_ang.values.min())

    # plot heatmap
    sum_resp = np.sum(responses, axis=2)
    # normalize by cols (for fixed receiver neuron, scale all responses
    # so that the strongest peaks to 1)
    sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T

    # scale to 0,1
    color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()

    cmap = plt.get_cmap(cmap_name)
    n_row, n_col, n_tp = responses.shape
    time = np.arange(n_tp)
    fig, axs = plt.subplots(n_row + 1, n_col + 1, figsize=figsize, sharey="row")
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            axs[rec, send].plot(time, responses[rec, send], color="k")
            axs[rec, send].spines["left"].set_visible(False)
            axs[rec, send].spines["bottom"].set_visible(False)
            axs[rec, send].set_xticks([])
            axs[rec, send].set_yticks([])
            axs[rec, send].axhline(0, color="k", lw=0.5)
            if rec == n_row - 1:
                axs[n_row, send].remove()  # Remove the original axis
                axs[n_row, send] = fig.add_subplot(n_row+1, n_col+1,
                                                   np.ravel_multi_index((n_row, send+1),(n_row + 1, n_col+1)),
                                                   polar=True)  # Add new polar axis

                axs[n_row, send].fill_between(
                    tuning.iloc[:, send].index,
                    np.zeros(len(tuning)),
                    tuning.iloc[:, send].values,
                    color=cmap_tun(color_tun[send]),
                    alpha=0.5,
                )
                axs[n_row, send].set_xticks([])
                axs[n_row, send].set_yticks([])

        axs[rec, send + 1].remove()  # Remove the original axis
        axs[rec, send + 1] = fig.add_subplot(n_row+1, n_col+1,
                                             np.ravel_multi_index((rec, send+1),(n_row+1, n_col+1)) + 1, polar=True)  # Add new polar axis

        axs[rec, send + 1].fill_between(
            tuning.iloc[:, rec].index,
            np.zeros(len(tuning)),
            tuning.iloc[:, rec].values,
            color=cmap_tun(color_tun[rec]),
            alpha=0.5,
        )
        axs[rec, send + 1].set_xticks([])
        axs[rec, send + 1].set_yticks([])
    axs[rec + 1, send + 1].set_xticks([])
    axs[rec + 1, send + 1].set_yticks([])
    axs[rec + 1, send + 1].spines["left"].set_visible(False)
    axs[rec + 1, send + 1].spines["bottom"].set_visible(False)
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            xlim = axs[rec, send].get_xlim()
            ylim = axs[rec, send].get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                alpha=alpha,
                color=cmap(color[rec, send]),
                zorder=1
            )
            axs[rec, send].add_patch(rect)
            axs[rec, send].set_xlim(xlim)
            axs[rec, send].set_ylim(ylim)
    axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
    axs[n_row, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)

    plt.suptitle("Pairwise Interaction", fontsize=fontsize)


def plot_history_window(neuron_count, interval, window_size_sec):
    bin_size = 1 / neuron_count.rate
    # define the count history window used for prediction
    history_interval = nap.IntervalSet(
        start=interval["start"][0], end=window_size_sec + interval["start"][0] - 0.001
    )

    # define the observed counts bin (the bin right after the history window)
    observed_count_interval = nap.IntervalSet(
        start=history_interval["end"], end=history_interval["end"] + bin_size
    )

    fig, _ = plt.subplots(1, 1, figsize=(8, 3.5))
    plt.step(
        neuron_count.restrict(interval).t, neuron_count.restrict(interval).d, where="post"
    )
    ylim = plt.ylim()
    plt.axvspan(
        history_interval["start"][0],
        history_interval["end"][0],
        *ylim,
        alpha=0.4,
        color="orange",
        label="history",
    )
    plt.axvspan(
        observed_count_interval["start"][0],
        observed_count_interval["end"][0],
        *ylim,
        alpha=0.4,
        color="tomato",
        label="predicted",
    )
    plt.ylim(ylim)
    plt.title("Spike Count Time Series")
    plt.xlabel("Time (sec)")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()


def plot_convolved_counts(counts, conv_spk, *epochs, figsize=(6.5, 4.5)):
    n_rows = len(epochs)
    fig, axs = plt.subplots(n_rows, 1, sharey="all", figsize=figsize)
    for row, ep in enumerate(epochs):
        axs[row].plot(conv_spk.restrict(ep))
        cnt_ep = counts.restrict(ep)
        axs[row].vlines(cnt_ep.t[cnt_ep.d > 0], -1, -0.1, "k", lw=2, label="spikes")

        if row == 0:
            axs[0].set_title("Convolved Counts")
            axs[0].legend()
        elif row == n_rows - 1:
            axs[row].set_xlabel("Time (sec)")


def plot_rates_and_smoothed_counts(counts, rate_dict,
                                   start=8819.4, end=8821, smooth_std=0.05, smooth_ws_scale=20):
    ep = nap.IntervalSet(start=start, end=end)
    fig = plt.figure()
    for key in  rate_dict:
        plt.plot(rate_dict[key].restrict(ep), label=key)

    idx_spikes = np.where(counts.restrict(ep).d > 0)[0]
    plt.vlines(counts.restrict(ep).t[idx_spikes], -8, -1, color="k")
    plt.plot(counts.smooth(smooth_std, size_factor=smooth_ws_scale).restrict(ep) * counts.rate, color="k", label="Smoothed spikes")
    plt.axhline(0, color="k")
    plt.xlabel("Time (sec)")
    plt.ylabel("Firing Rate (Hz)")
    plt.legend()


def plot_basis(n_basis_funcs=8, window_size_sec=0.8):
    fig = plt.figure()
    basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=n_basis_funcs)
    time, basis_kernels = basis.evaluate_on_grid(1000)
    time *= window_size_sec
    plt.plot(time, basis_kernels)
    plt.title("Log-stretched raised cosine basis")
    plt.xlabel("time (sec)")


def plot_current_history_features(current, features, basis, window_duration_sec,
                                  interval=nap.IntervalSet(462.77, 463)):
    fig, axes = plt.subplots(2, 3, sharey='row',  figsize=(8, 3.5))
    time, basis = basis.evaluate_on_grid(basis.window_size)
    time *= window_duration_sec
    current = current.restrict(interval)
    features = features.restrict(interval) / features.restrict(interval).max(0) * current.max()
    for ax in axes[1, :]:
        ax.plot(current, 'k--')
        ax.set_xlabel("Time (sec")
    axes[0, 0].plot(time, basis, alpha=.1)
    axes[0, 0].plot(time, basis[:, 0], 'C0', alpha=1)
    axes[0, 0].set_ylabel("Amplitude (A.U.)")
    axes[1, 0].plot(features[:,0])
    axes[1, 0].set_ylabel("Current")
    axes[0, 0].set_title("Feature 1")
    axes[1, 1].plot(features[:, -1], f'C{basis.shape[1]-1}')
    axes[0, 1].plot(time, basis, alpha=.1)
    axes[0, 1].plot(time, basis[:, -1], f'C{basis.shape[1]-1}', alpha=1)
    axes[0, 1].set_title(f"Feature {basis.shape[1]}")
    axes[0, 2].plot(time, basis)
    axes[1, 2].plot(features)
    axes[0, 2].set_title("All features")

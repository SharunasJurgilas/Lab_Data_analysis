"""
Various functions commonly required to look at and analyze data collected in MOT collisions experiments.
"""

from scipy.optimize import curve_fit
import numpy as np


def exp_model(t, tau, n0=1.0):
    """
    Single exponential decay model. Used to find, for example, the molecule MOT lifetime.
    Parameters
    ----------
    t: time coordinate
    tau: lifetime
    n0: signal at t=0
    """
    return n0 * np.exp(-t / tau)


def get_mot_life_times(data, delta_t, n_steps):
    """
    Function to fit CaF MOT lifetime data. Compatible with the experimental data structure in MOT collision
    experiments.
    Parameters
    ----------
    data: data numpy array with shape (n_experiments, n_time_steps * repetitions, image_size_x, image_size_y).
    delta_t: time delay between subsequent images when measuring the MOT lifetime.
    n_steps: number of time steps at which the MOT fluorescence is measured.

    Returns a numpy array of lifetime values. The shape of the returned array is (n_experiments, repetitions)
    """
    ts = np.arange(0, delta_t * n_steps, delta_t)
    taus = []
    for data_set in data:
        tau_data_set = np.array([])
        for experiment in data_set:
            tau_0 = np.where(experiment < 0.37)[0][0] * delta_t
            params, params_covariance = curve_fit(exp_model, ts, experiment, p0=[tau_0])
            tau_data_set = np.append(tau_data_set, params[0])
        taus.append(tau_data_set)
    return np.array(taus)


def tau_stats(taus):
    """
    Simple utility function which returns the mean and standard error values of the lifetime arrays.
    """
    n = taus.shape[1]
    return taus.mean(axis=1), taus.std(axis=1) / np.sqrt(n)


def taus_to_gamma(taus_with_rb, taus_without_rb, average=True):
    """
    Convert lifetimes to loss rates.
    Returns the average loss rates and corresponding standard errors if averages=True. Otherwise returns all of the
    loss rates.
    """
    if average:
        stats_with_rb = tau_stats(taus_with_rb)
        stats_without_rb = tau_stats(taus_without_rb)
        taus_with_rb_mean = stats_with_rb[0]
        taus_without_rb_mean = stats_without_rb[0]
        taus_with_rb_err = stats_with_rb[1]
        taus_without_rb_err = stats_without_rb[1]
        gamma = 1 / taus_with_rb_mean - 1 / taus_without_rb_mean
        gamma_err = np.sqrt(taus_without_rb_err ** 2 / taus_without_rb_mean ** 4
                             + taus_with_rb_err ** 2 / taus_with_rb_mean ** 4)
        return gamma, gamma_err
    else:
        return 1 / taus_with_rb - 1 / taus_without_rb

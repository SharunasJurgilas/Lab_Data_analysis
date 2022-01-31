"""
Functions required to estimate the cloud size and position.
"""
import numpy as np
from scipy.optimize import curve_fit


def gauss1d(x, a, b, xc, sigma):
    """
    Evaluates a 1D Gaussian.
    Parameters
    ----------
    x : position
    a : amplitude
    b : offset
    xc : location of peak (mean position)
    sigma : rms width
    """
    return a + b * np.exp(-(x - xc) ** 2 / (2 * sigma ** 2))


def gauss2d(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Evaluates a 2D Gaussian.
    """
    x, y = pos
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                            + c * ((y - yo) ** 2)))
    return g.ravel()


def help_gauss(coordinates, data):
    """
    Returns initial parameter guesses for a 1D Gaussian fit.
    """
    pixel_number = np.shape(data)[0]
    b0=np.amax(data)
    a0=np.amin(data)
    x0 = data.argmax() * coordinates.max() / pixel_number
    sigma_0 = 0.5 * (len(data[(data <= b0) & (data >= 0.37 * b0)])) * coordinates.max() / pixel_number
    return [a0, b0, x0, sigma_0]


def fit_gauss_1d(coordinates, data):
    """
    Returns 1D Gaussian fit parameters.
    Parameters
    ----------
    coordinates: coordinate values matching the data.
    data: numpy array containing the data to be fitted to. Should be of shape (n_experiments, n_repetitions,
    array length).
    """
    d1 = data.shape[1]
    parameter_container = []
    for experiment in data:
        experiment_params = np.array([])
        for rep in experiment:
            p0 = help_gauss(coordinates, rep)
            params, params_covariance = curve_fit(gauss1d, coordinates, rep, p0=p0)
            experiment_params = np.append(experiment_params, np.array(params))
        parameter_container.append(experiment_params)
    return np.array(parameter_container).reshape(-1, d1, 4)


def fit_gauss_2d(xy, data, guess_x, guess_y):
    """
    Returns 2D Gaussian fit parameters.
    Parameters
    ----------
    xy: x-y mesh. Can make with np.meshgrid(x, y), where x is the x-coordinates and y is the y-coordinates.
    data: numpy array containing the data to be fitted to. Should be of shape (n_experiments, n_repetitions,
    image dimension along x, image dimension along y).
    guess_x: array containing initial guesses of  centre position and rms radius along x. Dimensions should
    be (n_experiments, n_repetitions, 1, 1).
    guess_y: array containing initial guesses of  centre position and rms radius along y. Dimensions should
    be (n_experiments, n_repetitions, 1, 1).
    """
    d1 = data.shape[1]
    parameter_container = []
    for experiment, ex, ey in zip(data, guess_x, guess_y):
        experiment_params = np.array([])
        for rep, rx, ry in zip(experiment, ex, ey):
            im_for_fit = rep.ravel()
            x, y = xy
            amplitude0 = im_for_fit.max()
            x00 = rx[0]
            y00 = ry[0]
            sigma_x0 = rx[1]
            sigma_y0 = ry[1]
            theta = 0.0
            offset0 = 0.0
            p0 = [amplitude0, x00, y00, sigma_x0, sigma_y0, theta, offset0]
            params, pcov = curve_fit(gauss2d, (x, y), im_for_fit, p0=p0)
            experiment_params = np.append(experiment_params, np.array(params))
        parameter_container.append(experiment_params)
    return np.array(parameter_container).reshape(-1, d1, 7)
